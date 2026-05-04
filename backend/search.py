import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

from distiller_backend import DistillerBackend
from layer_inventory import build_layer_inventory, load_model_for_inventory
from skin_distiller_backend import SkinDistillerBackend


DEFAULT_SKIN_NINE_CHECKPOINT_CANDIDATES = [
    "/home/mussa/skin-fairness-ptq/backend/skin_checkpoints/exp_server_e60_s42/fitz17k_nine_resnet18_best.pth",
    "/home/mussa/skin-fairness-ptq/backend/skin_checkpoints/fitz17k_nine_resnet18_best.pth",
]


def _pick_existing_checkpoint(candidates):
    for path in candidates:
        p = Path(path)
        if p.exists():
            return str(p)
    return None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def parse_bits(bits_str):
    bits = [int(x.strip()) for x in bits_str.split(",") if x.strip()]
    if not bits:
        raise ValueError("bit options must not be empty")
    return sorted(set(bits), reverse=True)


def next_lower_bit(curr_bit, bit_options_desc):
    idx = bit_options_desc.index(curr_bit)
    if idx + 1 >= len(bit_options_desc):
        return None
    return bit_options_desc[idx + 1]


def _percentile(values, q):
    if np is not None:
        return float(np.percentile(values, q))
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    s = sorted(float(v) for v in values)
    pos = (len(s) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + frac * (s[hi] - s[lo])


def compute_bands(values, banding):
    if banding == "fixed":
        return {
            "banding": "fixed",
            "cutpoints": {"p25": 0.010, "p50": 0.020, "p75": 0.030},
        }
    p25 = _percentile(values, 25)
    p50 = _percentile(values, 50)
    p75 = _percentile(values, 75)
    return {
        "banding": "percentile",
        "cutpoints": {"p25": float(p25), "p50": float(p50), "p75": float(p75)},
    }


def sensitivity_group(drop, cutpoints):
    if drop <= cutpoints["p25"]:
        return "very_low"
    if drop <= cutpoints["p50"]:
        return "low"
    if drop <= cutpoints["p75"]:
        return "moderate"
    return "high"


def theoretical_size_bits(assignment, params_by_layer):
    total = 0
    for layer, bits in assignment.items():
        total += int(params_by_layer[layer]) * int(bits)
    return int(total)


def bit_distribution(assignment):
    dist = {}
    for _, bit in assignment.items():
        key = str(int(bit))
        dist[key] = dist.get(key, 0) + 1
    return dict(sorted(dist.items(), key=lambda kv: int(kv[0]), reverse=True))


def setup_logger(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    logger = logging.getLogger("mp_search")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def pick_probe_layers(layers, max_layers, mode):
    if max_layers <= 0 or max_layers >= len(layers):
        return list(layers)
    if mode == "head":
        return list(layers[:max_layers])
    if mode == "tail":
        return list(layers[-max_layers:])
    if mode == "stride":
        stride = max(1, len(layers) // max_layers)
        picked = layers[::stride]
        return list(picked[:max_layers])
    return list(layers[:max_layers])


def measure_sensitivities(
    backend,
    layers,
    probe_layers,
    base_wts,
    base_acts,
    baseline_top1,
    target,
    probe_bit,
    logger,
):
    sensitivities = {}
    for idx, layer in enumerate(probe_layers, start=1):
        trial_wts = dict(base_wts)
        trial_acts = dict(base_acts)
        if target == "wts":
            trial_wts[layer] = probe_bit
        else:
            trial_acts[layer] = probe_bit
        metrics = backend.evaluate_quantized(assignment_wts=trial_wts, assignment_acts=trial_acts)
        drop = baseline_top1 - metrics["top1"]
        sensitivities[layer] = {
            "drop": drop,
            "top1": metrics["top1"],
            "top5": metrics["top5"],
            "loss": metrics["loss"],
        }
        logger.info("[%s %d/%d] %s probe_top1=%.3f drop=%.3f",
                    target, idx, len(probe_layers), layer, metrics["top1"], drop)

    if len(probe_layers) < len(layers):
        median = _percentile([v["drop"] for v in sensitivities.values()], 50) if sensitivities else 0.0
        for layer in layers:
            if layer not in sensitivities:
                sensitivities[layer] = {
                    "drop": float(median),
                    "top1": float("nan"),
                    "top5": float("nan"),
                    "loss": float("nan"),
                }
    return sensitivities


def write_sensitivities_csv(path, layers, sensitivities):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "sensitivity_drop", "probe_top1", "probe_top5", "probe_loss"])
        for layer in layers:
            s = sensitivities[layer]
            writer.writerow([
                layer,
                f"{s['drop']:.6f}",
                f"{s['top1']:.6f}" if s["top1"] == s["top1"] else "",
                f"{s['top5']:.6f}" if s["top5"] == s["top5"] else "",
                f"{s['loss']:.6f}" if s["loss"] == s["loss"] else "",
            ])


def build_groups(layers, sensitivities, banding):
    values = [sensitivities[layer]["drop"] for layer in layers]
    banding_meta = compute_bands(values, banding)
    cutpoints = banding_meta["cutpoints"]
    groups = {"very_low": [], "low": [], "moderate": [], "high": []}
    for layer in layers:
        groups[sensitivity_group(sensitivities[layer]["drop"], cutpoints)].append(layer)
    return {
        "banding": banding_meta["banding"],
        "cutpoints": {
            "p25": round(cutpoints["p25"], 6),
            "p50": round(cutpoints["p50"], 6),
            "p75": round(cutpoints["p75"], 6),
        },
        "groups": groups,
    }


def progressive_group_search(backend, assignment_wts, assignment_acts, groups_payload, target, candidate_bits, budget_top1, logger):
    groups = groups_payload["groups"]
    ordered_groups = ["very_low", "low", "moderate", "high"]
    out_wts = dict(assignment_wts)
    out_acts = dict(assignment_acts)
    for group_name in ordered_groups:
        layers = groups[group_name]
        if not layers:
            continue
        logger.info("%s group %s size=%d", target, group_name, len(layers))
        for bit in candidate_bits:
            trial_wts = dict(out_wts)
            trial_acts = dict(out_acts)
            for layer in layers:
                if target == "wts":
                    trial_wts[layer] = bit
                else:
                    trial_acts[layer] = bit
            metrics = backend.evaluate_quantized(assignment_wts=trial_wts, assignment_acts=trial_acts)
            ok = metrics["top1"] >= budget_top1
            logger.info("  %s try bit=%d -> top1=%.3f (%s)", target, bit, metrics["top1"], "accept" if ok else "reject")
            if ok:
                out_wts, out_acts = trial_wts, trial_acts
    return out_wts, out_acts


def greedy_refinement(backend, assignment_wts, assignment_acts, layers_sorted, target, bit_options, budget_top1, passes, logger):
    out_wts = dict(assignment_wts)
    out_acts = dict(assignment_acts)
    for p in range(passes):
        logger.info("%s refinement pass %d/%d", target, p + 1, passes)
        changed = 0
        for layer in layers_sorted:
            curr_bit = out_wts[layer] if target == "wts" else out_acts[layer]
            lower_bit = next_lower_bit(curr_bit, bit_options)
            if lower_bit is None:
                continue
            trial_wts = dict(out_wts)
            trial_acts = dict(out_acts)
            if target == "wts":
                trial_wts[layer] = lower_bit
            else:
                trial_acts[layer] = lower_bit
            metrics = backend.evaluate_quantized(assignment_wts=trial_wts, assignment_acts=trial_acts)
            if metrics["top1"] >= budget_top1:
                out_wts, out_acts = trial_wts, trial_acts
                changed += 1
                logger.info("  %s %s: %d -> %d accepted (top1=%.3f)", target, layer, curr_bit, lower_bit, metrics["top1"])
            else:
                logger.info("  %s %s: %d -> %d rejected (top1=%.3f)", target, layer, curr_bit, lower_bit, metrics["top1"])
        if changed == 0:
            logger.info("%s refinement converged early", target)
            break
    return out_wts, out_acts


def main():
    parser = argparse.ArgumentParser(description="Sensitivity-aware mixed-precision search using Distiller backend")
    parser.add_argument("--task", choices=["cifar", "skin_fitzpatrick"], default="cifar")
    parser.add_argument("--data-dir", default="/home/mussa/skin-fairness-ptq/data")
    parser.add_argument("--checkpoint", default="/home/mussa/skin-fairness-ptq/distiller/examples/classifier_compression/r56_train_best.pth.tar")
    parser.add_argument("--arch", default="resnet56_cifar")
    parser.add_argument("--skin-csv", default="/home/mussa/skin_fairness_project/data/fitzpatrick17k.csv")
    parser.add_argument("--skin-image-dir", default="/home/mussa/fitzpatrick17k_data/data/finalfitz17k")
    parser.add_argument("--skin-label-space", choices=["auto", "binary", "nine"], default="nine")
    parser.add_argument("--skin-eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--skin-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--bit-options-wts", default="8,7,6,5,4")
    parser.add_argument("--bit-options-acts", default="8,7,6")
    parser.add_argument("--allow-wts-very-low", action="store_true", help="Allow 3/2 in weights search")
    parser.add_argument("--allow-a4", action="store_true", help="Allow activation=4 in activation search")
    parser.add_argument("--probe-bit-wts", type=int, default=4)
    parser.add_argument("--probe-bit-acts", type=int, default=6)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--refinement-passes", type=int, default=2)
    parser.add_argument("--banding", choices=["fixed", "percentile"], default="percentile")
    parser.add_argument("--search-mode", choices=["weights", "weights_acts"], default="weights_acts")
    parser.add_argument("--act-skip-first-layers", type=int, default=8)
    parser.add_argument("--max-probe-layers", type=int, default=0)
    parser.add_argument("--probe-mode", choices=["head", "tail", "stride"], default="head")
    parser.add_argument("--backend", choices=["subprocess", "inprocess"], default="subprocess")
    parser.add_argument("--distiller-dir", default="/home/mussa/skin-fairness-ptq/distiller")
    parser.add_argument("--python-bin", default="/home/mussa/skin-fairness-ptq/.venv/bin/python")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    if args.task == "skin_fitzpatrick" and args.arch == "resnet56_cifar":
        args.arch = "resnet18"
    if args.task == "skin_fitzpatrick" and args.checkpoint == parser.get_default("checkpoint"):
        picked = _pick_existing_checkpoint(DEFAULT_SKIN_NINE_CHECKPOINT_CANDIDATES)
        if picked is not None:
            args.checkpoint = picked

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"/home/mussa/skin-fairness-ptq/backend/results/run_{ts}")
    logger = setup_logger(output_dir)
    logger.info("Output directory: %s", output_dir)

    w_bits = parse_bits(args.bit_options_wts)
    if args.allow_wts_very_low:
        for b in [3, 2]:
            if b not in w_bits:
                w_bits.append(b)
        w_bits = sorted(set(w_bits), reverse=True)
    a_bits = parse_bits(args.bit_options_acts)
    if args.allow_a4 and 4 not in a_bits:
        a_bits.append(4)
        a_bits = sorted(set(a_bits), reverse=True)

    if 8 not in w_bits or 8 not in a_bits:
        raise ValueError("Both weight and activation bit options must include 8")
    if args.probe_bit_wts not in w_bits:
        raise ValueError("--probe-bit-wts must be in --bit-options-wts")
    if args.probe_bit_acts not in a_bits:
        raise ValueError("--probe-bit-acts must be in --bit-options-acts")

    if args.task == "cifar":
        model = load_model_for_inventory(args.arch, args.distiller_dir)
        inventory = build_layer_inventory(model)
    else:
        skin_backend_for_inventory = SkinDistillerBackend(
            data_csv=args.skin_csv,
            image_dir=args.skin_image_dir,
            checkpoint=args.checkpoint,
            eval_batches=args.eval_batches,
            output_dir=output_dir,
            seed=args.seed,
            batch_size=args.skin_batch_size,
            eval_split=args.skin_eval_split,
            distiller_dir=args.distiller_dir,
            label_space=args.skin_label_space,
        )
        model = skin_backend_for_inventory.load_model_for_inventory()
        inventory = build_layer_inventory(model)
    layers = [entry["name"] for entry in inventory["layers"]]
    params_by_layer = {entry["name"]: int(entry["weight_params"]) for entry in inventory["layers"]}
    (output_dir / "layer_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    logger.info("Quantizable layers: %d", len(layers))

    if args.task == "cifar":
        backend = DistillerBackend(
            data_dir=args.data_dir,
            checkpoint=args.checkpoint,
            arch=args.arch,
            eval_batches=args.eval_batches,
            output_dir=output_dir,
            python_bin=args.python_bin,
            distiller_dir=args.distiller_dir,
            backend=args.backend,
        )
    else:
        backend = SkinDistillerBackend(
            data_csv=args.skin_csv,
            image_dir=args.skin_image_dir,
            checkpoint=args.checkpoint,
            eval_batches=args.eval_batches,
            output_dir=output_dir,
            seed=args.seed,
            batch_size=args.skin_batch_size,
            eval_split=args.skin_eval_split,
            distiller_dir=args.distiller_dir,
            label_space=args.skin_label_space,
        )
    cache_path = output_dir / "eval_cache.json"
    backend.load_cache(cache_path)

    baseline = backend.evaluate_baseline()
    baseline_top1 = baseline["top1"]
    (output_dir / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    logger.info("Stage0 baseline: Top1=%.3f", baseline_top1)

    assignment_wts = {layer: 8 for layer in layers}
    assignment_acts = {layer: 8 for layer in layers}
    ptq_default = backend.evaluate_quantized(assignment_wts=assignment_wts, assignment_acts=assignment_acts)
    (output_dir / "ptq_default.json").write_text(json.dumps(ptq_default, indent=2) + "\n")
    logger.info("Stage0 PTQ default W8/A8: Top1=%.3f", ptq_default["top1"])

    budget_top1 = baseline_top1 - args.epsilon
    probe_layers_w = pick_probe_layers(layers, args.max_probe_layers, args.probe_mode)

    logger.info("Stage1 weights-only search; probe layers=%d", len(probe_layers_w))
    sens_w = measure_sensitivities(
        backend,
        layers,
        probe_layers_w,
        assignment_wts,
        assignment_acts,
        baseline_top1,
        target="wts",
        probe_bit=args.probe_bit_wts,
        logger=logger,
    )
    write_sensitivities_csv(output_dir / "sensitivities_weights.csv", layers, sens_w)
    groups_w = build_groups(layers, sens_w, args.banding)
    logger.info("Weights groups sizes: very_low=%d low=%d moderate=%d high=%d",
                len(groups_w["groups"]["very_low"]), len(groups_w["groups"]["low"]),
                len(groups_w["groups"]["moderate"]), len(groups_w["groups"]["high"]))
    candidate_w = [b for b in w_bits if b < 8]
    assignment_wts, assignment_acts = progressive_group_search(
        backend,
        assignment_wts,
        assignment_acts,
        groups_w,
        target="wts",
        candidate_bits=candidate_w,
        budget_top1=budget_top1,
        logger=logger,
    )
    layers_sorted_w = sorted(layers, key=lambda name: sens_w[name]["drop"])
    assignment_wts, assignment_acts = greedy_refinement(
        backend,
        assignment_wts,
        assignment_acts,
        layers_sorted_w,
        target="wts",
        bit_options=w_bits,
        budget_top1=budget_top1,
        passes=args.refinement_passes,
        logger=logger,
    )

    sens_a = {}
    groups_a = {"banding": args.banding, "cutpoints": {"p25": 0.0, "p50": 0.0, "p75": 0.0}, "groups": {
        "very_low": [], "low": [], "moderate": [], "high": []
    }}

    if args.search_mode == "weights_acts":
        logger.info("Stage2 activation search on top of weights assignment")
        eligible_act_layers = layers[args.act_skip_first_layers:]
        probe_layers_a = pick_probe_layers(eligible_act_layers, args.max_probe_layers, args.probe_mode)
        logger.info("Activation skip-first-layers=%d eligible=%d probe=%d",
                    args.act_skip_first_layers, len(eligible_act_layers), len(probe_layers_a))
        sens_a = measure_sensitivities(
            backend,
            eligible_act_layers,
            probe_layers_a,
            assignment_wts,
            assignment_acts,
            baseline_top1,
            target="acts",
            probe_bit=args.probe_bit_acts,
            logger=logger,
        )
        for layer in layers[:args.act_skip_first_layers]:
            sens_a[layer] = {"drop": 999.0, "top1": float("nan"), "top5": float("nan"), "loss": float("nan")}
        write_sensitivities_csv(output_dir / "sensitivities_activations.csv", layers, sens_a)

        groups_a = build_groups(eligible_act_layers, sens_a, args.banding)
        logger.info("Activations groups sizes: very_low=%d low=%d moderate=%d high=%d",
                    len(groups_a["groups"]["very_low"]), len(groups_a["groups"]["low"]),
                    len(groups_a["groups"]["moderate"]), len(groups_a["groups"]["high"]))

        candidate_a = [b for b in a_bits if b < 8]
        assignment_wts, assignment_acts = progressive_group_search(
            backend,
            assignment_wts,
            assignment_acts,
            groups_a,
            target="acts",
            candidate_bits=candidate_a,
            budget_top1=budget_top1,
            logger=logger,
        )
        layers_sorted_a = sorted(eligible_act_layers, key=lambda name: sens_a[name]["drop"])
        assignment_wts, assignment_acts = greedy_refinement(
            backend,
            assignment_wts,
            assignment_acts,
            layers_sorted_a,
            target="acts",
            bit_options=a_bits,
            budget_top1=budget_top1,
            passes=args.refinement_passes,
            logger=logger,
        )
    else:
        write_sensitivities_csv(output_dir / "sensitivities_activations.csv", layers, {
            layer: {"drop": 0.0, "top1": float("nan"), "top5": float("nan"), "loss": float("nan")} for layer in layers
        })

    final_metrics = backend.evaluate_quantized(assignment_wts=assignment_wts, assignment_acts=assignment_acts)
    (output_dir / "assignment_weights.json").write_text(json.dumps(assignment_wts, indent=2) + "\n")
    (output_dir / "assignment_activations.json").write_text(json.dumps(assignment_acts, indent=2) + "\n")
    groups_payload = {"weights": groups_w, "activations": groups_a}
    (output_dir / "groups.json").write_text(json.dumps(groups_payload, indent=2) + "\n")

    fp32_weight_bits = sum(params_by_layer.values()) * 32
    weight_bits = theoretical_size_bits(assignment_wts, params_by_layer)
    activation_bits_proxy = theoretical_size_bits(assignment_acts, params_by_layer)
    combined_bits_proxy = weight_bits + activation_bits_proxy
    compression_ratio_weights = fp32_weight_bits / weight_bits if weight_bits else 0.0

    final = {
        "baseline_top1": baseline_top1,
        "ptq_default_top1": ptq_default["top1"],
        "final_top1": final_metrics["top1"],
        "drop_vs_baseline": baseline_top1 - final_metrics["top1"],
        "budget_epsilon": args.epsilon,
        "within_budget": final_metrics["top1"] >= budget_top1,
        "arch": args.arch,
        "task": args.task,
        "skin_label_space": args.skin_label_space,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "seed": args.seed,
        "search_mode": args.search_mode,
        "eval_batches": args.eval_batches,
        "banding": args.banding,
        "weight_bit_options": w_bits,
        "activation_bit_options": a_bits,
        "probe_bit_wts": args.probe_bit_wts,
        "probe_bit_acts": args.probe_bit_acts,
        "refinement_passes": args.refinement_passes,
        "weights_theoretical_bits": weight_bits,
        "fp32_weights_bits": fp32_weight_bits,
        "compression_ratio_vs_fp32_weights": compression_ratio_weights,
        "activations_theoretical_bits_proxy": activation_bits_proxy,
        "combined_theoretical_bits_proxy": combined_bits_proxy,
        "weights_bit_distribution": bit_distribution(assignment_wts),
        "activations_bit_distribution": bit_distribution(assignment_acts),
        "num_evals_executed": backend.eval_count,
    }
    for metric_key in [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "tp",
        "tn",
        "fp",
        "fn",
        "n_samples",
        "num_classes",
        "label_space",
        "class_names",
        "eval_split",
        "unfairness_score",
        "group_accuracies",
    ]:
        if metric_key in baseline:
            final[f"baseline_{metric_key}"] = baseline[metric_key]
        if metric_key in final_metrics:
            final[f"final_{metric_key}"] = final_metrics[metric_key]
    (output_dir / "final.json").write_text(json.dumps(final, indent=2) + "\n")
    backend.save_cache(cache_path)

    logger.info("Final Top1=%.3f drop=%.3f within_budget=%s", final["final_top1"], final["drop_vs_baseline"], final["within_budget"])
    logger.info("Weights compression=%.3fx evals=%d", final["compression_ratio_vs_fp32_weights"], final["num_evals_executed"])
    logger.info("Done")


if __name__ == "__main__":
    main()
