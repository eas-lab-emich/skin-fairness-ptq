"""
Greedy Fairness-Aware Precision Allocation
===========================================

Algorithm (Professor's greedy idea):
  1. Start with all layers at low precision (e.g. W4A4)
  2. For each layer, temporarily increase precision one step and measure
     improvement in worst-group accuracy (WGA = min accuracy over Fitzpatrick groups).
     Compute ratio = ΔWGA / Δmemory_bits.
  3. Upgrade the layer+component with the best ratio.
  4. Repeat until the memory budget is exhausted (or no improvement possible).

Key metric: WGA (worst-group accuracy) = min(group_accuracies.values())
  — minimax fairness criterion: directly optimize the least-served group.

Imports backend from ../backend/ via sys.path (Option A).
Historical backend/ code is left completely untouched.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# ── Import shared utilities from the existing project (no code duplication) ──
_BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
sys.path.insert(0, _BACKEND_DIR)

from skin_distiller_backend import SkinDistillerBackend  # noqa: E402
from layer_inventory import build_layer_inventory         # noqa: E402

# ── Default checkpoint candidates (resolved per arch in main()) ──────────────
_DEFAULT_CHECKPOINT_CANDIDATES_RESNET18 = [
    "/home/mussa/skin-fairness-ptq/backend/skin_checkpoints/exp_server_e60_s42/fitz17k_nine_resnet18_best.pth",
    "/home/mussa/skin-fairness-ptq/backend/skin_checkpoints/fitz17k_nine_resnet18_best.pth",
]
_CKPT_ROOT = "/home/mussa/skin-fairness-ptq/backend/skin_checkpoints"


def _default_checkpoint_candidates(arch, label_space="nine"):
    """Return ordered list of default checkpoint paths to try for a given arch."""
    root = Path(_CKPT_ROOT)
    return [
        str(root / f"fitz17k_{label_space}_{arch}_best.pth"),
        # Legacy location used for the original resnet18 run
        str(root / f"exp_server_e60_s42/fitz17k_{label_space}_{arch}_best.pth"),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def _pick_existing_checkpoint(candidates):
    for path in candidates:
        if Path(path).exists():
            return str(path)
    return None


def setup_logger(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    logger = logging.getLogger("greedy_fairness")
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


def parse_bits(bits_str):
    bits = [int(x.strip()) for x in bits_str.split(",") if x.strip()]
    if not bits:
        raise ValueError("bit options must not be empty")
    return sorted(set(bits), reverse=True)  # descending


def compute_wga(metrics):
    """Worst-group accuracy = min accuracy across Fitzpatrick groups 1-6."""
    ga = metrics.get("group_accuracies")
    if not ga:
        return 0.0
    return min(float(v) for v in ga.values())


def next_higher_bit(curr_bit, bit_options_desc):
    """Return the next higher bit from a descending list, or None if already at max.

    E.g. with [8,7,6,5,4]: 4→5, 5→6, ..., 7→8, 8→None.
    Mirror of next_lower_bit in the original search.py.
    """
    idx = bit_options_desc.index(curr_bit)
    if idx == 0:
        return None
    return bit_options_desc[idx - 1]


def theoretical_size_bits(assignment, params_by_layer):
    """Total bit count for one assignment dict: sum(params × bits)."""
    return sum(int(params_by_layer[l]) * int(b) for l, b in assignment.items())


def compute_current_budget_bits(assignment_wts, assignment_acts, params_by_layer, budget_mode):
    """Tracked budget bits under the chosen budget mode.

    'weights'        → only weight bits count toward the budget.
    'combined_proxy' → weight bits + activation bits (using weight_params as proxy).
    """
    total = 0
    for layer, params in params_by_layer.items():
        wbits = int(assignment_wts.get(layer, 8))
        if budget_mode == "combined_proxy":
            abits = int(assignment_acts.get(layer, 8))
            total += params * (wbits + abits)
        else:
            total += params * wbits
    return int(total)


def bit_distribution(assignment):
    dist = {}
    for _, bit in assignment.items():
        key = str(int(bit))
        dist[key] = dist.get(key, 0) + 1
    return dict(sorted(dist.items(), key=lambda kv: int(kv[0]), reverse=True))


# ═══════════════════════════════════════════════════════════════════════════════
# Core algorithm
# ═══════════════════════════════════════════════════════════════════════════════

def greedy_fairness_search(
    backend,
    layers,
    params_by_layer,
    w_bits,
    a_bits,
    start_bits_wts,
    start_bits_acts,
    budget_ratio,
    budget_mode,
    upgrade_acts,
    plateau_patience,
    logger,
    max_ud=None,
):
    """Greedy fairness-aware mixed-precision search.

    Starts all layers at (start_bits_wts, start_bits_acts) and iteratively
    upgrades the single layer+component that yields the best ΔWGA/Δbits ratio,
    stopping when the budget is exhausted, all layers are at max, or WGA has
    not improved for plateau_patience+1 consecutive iterations.

    Returns:
        assignment_wts  : dict[layer → int]
        assignment_acts : dict[layer → int]
        history         : list[dict]  — one entry per completed upgrade
    """
    # ── Initialise all layers at start precision ──────────────────────────────
    assignment_wts  = {layer: start_bits_wts  for layer in layers}
    assignment_acts = {layer: start_bits_acts for layer in layers}

    # ── Budget arithmetic ─────────────────────────────────────────────────────
    fp32_weight_bits = sum(params_by_layer.values()) * 32
    if budget_mode == "combined_proxy":
        fp32_ref_bits = fp32_weight_bits * 2   # wts + acts proxy at fp32
    else:
        fp32_ref_bits = fp32_weight_bits
    budget_max_bits = int(fp32_ref_bits / budget_ratio)

    # ── Evaluate starting state ───────────────────────────────────────────────
    logger.info(
        "Greedy search: evaluating W%dA%d starting point (cold cache — may take a few minutes)...",
        start_bits_wts, start_bits_acts,
    )
    start_metrics = backend.evaluate_quantized(
        assignment_wts=assignment_wts,
        assignment_acts=assignment_acts,
    )
    current_wga   = compute_wga(start_metrics)
    current_bits  = compute_current_budget_bits(assignment_wts, assignment_acts, params_by_layer, budget_mode)
    current_cr    = fp32_weight_bits / theoretical_size_bits(assignment_wts, params_by_layer)

    logger.info(
        "Start | W%dA%d | WGA=%.4f | UD=%.4f | top1=%.3f | macro_f1=%.4f | "
        "budget_bits=%d / %d | compression=%.2fx",
        start_bits_wts, start_bits_acts,
        current_wga,
        start_metrics.get("unfairness_score", float("nan")),
        start_metrics["top1"],
        start_metrics.get("macro_f1", float("nan")),
        current_bits, budget_max_bits, current_cr,
    )
    logger.info(
        "Budget: fp32_ref_bits=%d, max_bits=%d (ratio=%.1fx). "
        "First iteration probes all %d candidates (subsequent iterations mostly cache hits).",
        fp32_ref_bits, budget_max_bits, budget_ratio, len(layers) * (1 + int(upgrade_acts)),
    )

    history = []
    all_candidates_log = []
    plateau_streak = 0
    iteration = 0

    # ── Main greedy loop ──────────────────────────────────────────────────────
    while True:
        iteration += 1

        # Build all feasible one-step upgrade candidates
        feasible = []
        for layer in layers:
            params = params_by_layer[layer]

            # Weight upgrade candidate
            curr_wbits = int(assignment_wts[layer])
            next_wbits = next_higher_bit(curr_wbits, w_bits)
            if next_wbits is not None:
                delta_bits = params * (next_wbits - curr_wbits)
                within = (current_bits + delta_bits) <= budget_max_bits
                feasible.append({
                    "layer": layer,
                    "component": "wts",
                    "old_bits": curr_wbits,
                    "new_bits": next_wbits,
                    "delta_bits": int(delta_bits),
                    "within_budget": within,
                })

            # Activation upgrade candidate
            if upgrade_acts:
                curr_abits = int(assignment_acts[layer])
                next_abits = next_higher_bit(curr_abits, a_bits)
                if next_abits is not None:
                    if budget_mode == "combined_proxy":
                        delta_bits_act = params * (next_abits - curr_abits)
                    else:
                        delta_bits_act = 0   # acts are free in weights-only budget mode
                    within = (current_bits + delta_bits_act) <= budget_max_bits
                    feasible.append({
                        "layer": layer,
                        "component": "acts",
                        "old_bits": curr_abits,
                        "new_bits": next_abits,
                        "delta_bits": int(delta_bits_act),
                        "within_budget": within,
                    })

        feasible_within = [c for c in feasible if c["within_budget"]]

        if not feasible_within:
            logger.info("Iter %d: no feasible upgrades within budget. Stopping (reason: budget_exhausted).", iteration)
            history_stopping_reason = "budget_exhausted"
            break

        # Probe every feasible candidate
        candidate_results = []
        for c in feasible_within:
            trial_wts  = dict(assignment_wts)
            trial_acts = dict(assignment_acts)
            if c["component"] == "wts":
                trial_wts[c["layer"]]  = c["new_bits"]
            else:
                trial_acts[c["layer"]] = c["new_bits"]

            metrics = backend.evaluate_quantized(
                assignment_wts=trial_wts,
                assignment_acts=trial_acts,
            )
            probe_wga = compute_wga(metrics)
            delta_wga = probe_wga - current_wga
            ud_score  = metrics.get("unfairness_score", 0.0)

            # ratio = ΔWGA / Δbits.
            # When delta_bits == 0 (acts in weights-only budget mode), the upgrade
            # costs no budget — rank by ΔWGA directly via a large sentinel.
            if c["delta_bits"] > 0:
                ratio = delta_wga / c["delta_bits"]
            else:
                ratio = delta_wga * 1e9

            ud_rejected = max_ud is not None and ud_score > max_ud

            # Always log every evaluated candidate for Pareto front analysis
            all_candidates_log.append({
                "iteration": iteration,
                "layer": c["layer"],
                "component": c["component"],
                "old_bits": c["old_bits"],
                "new_bits": c["new_bits"],
                "delta_wga": delta_wga,
                "delta_bits": c["delta_bits"],
                "ratio": ratio,
                "probe_wga": probe_wga,
                "ud_score": ud_score,
                "top1": metrics["top1"],
                "macro_f1": metrics.get("macro_f1"),
                "group_accuracies": metrics.get("group_accuracies"),
                "rejected_ud_ceiling": ud_rejected,
            })

            if ud_rejected:
                logger.info(
                    "  Iter %d REJECT(UD) | %s [%s] %d→%d | UD=%.4f > max_ud=%.4f",
                    iteration, c["layer"], c["component"],
                    c["old_bits"], c["new_bits"], ud_score, max_ud,
                )
                continue

            candidate_results.append({
                **c,
                "probe_wga": probe_wga,
                "delta_wga": delta_wga,
                "ratio": ratio,
                "probe_metrics": metrics,
            })
            logger.info(
                "  Iter %d probe | %s [%s] %d→%d | ΔWGA=%.5f | UD=%.4f | Δbits=%d | ratio=%.3e",
                iteration, c["layer"], c["component"],
                c["old_bits"], c["new_bits"],
                delta_wga, ud_score, c["delta_bits"], ratio,
            )

        if not candidate_results:
            logger.info(
                "Iter %d: all %d candidates rejected by UD ceiling (max_ud=%.4f). Stopping.",
                iteration, len(feasible_within), max_ud,
            )
            history_stopping_reason = "ud_ceiling_exhausted"
            break

        # Select best candidate: max ratio, break ties by smaller delta_bits
        best = max(candidate_results, key=lambda r: (r["ratio"], -r["delta_bits"]))

        # Plateau check
        if best["delta_wga"] <= 0.0:
            plateau_streak += 1
            logger.info(
                "Iter %d: best ΔWGA=%.5f ≤ 0 (plateau_streak=%d/%d)",
                iteration, best["delta_wga"], plateau_streak, plateau_patience + 1,
            )
            if plateau_streak > plateau_patience:
                logger.info("Plateau patience exhausted. Stopping (reason: plateau).")
                history_stopping_reason = "plateau"
                break
        else:
            plateau_streak = 0

        # Commit the upgrade
        if best["component"] == "wts":
            assignment_wts[best["layer"]] = best["new_bits"]
        else:
            assignment_acts[best["layer"]] = best["new_bits"]

        new_bits = compute_current_budget_bits(
            assignment_wts, assignment_acts, params_by_layer, budget_mode
        )
        new_cr = fp32_weight_bits / theoretical_size_bits(assignment_wts, params_by_layer)

        logger.info(
            "Iter %d SELECT | %s [%s] %d→%d | ΔWGA=%.5f | WGA %.4f→%.4f | "
            "UD=%.4f | top1=%.3f | macro_f1=%.4f | bits=%d | compression=%.2fx",
            iteration,
            best["layer"], best["component"],
            best["old_bits"], best["new_bits"],
            best["delta_wga"], current_wga, best["probe_wga"],
            best["probe_metrics"].get("unfairness_score", float("nan")),
            best["probe_metrics"]["top1"],
            best["probe_metrics"].get("macro_f1", float("nan")),
            new_bits, new_cr,
        )

        # Record iteration in history
        history.append({
            "iteration": iteration,
            "layer": best["layer"],
            "component": best["component"],
            "old_bits": best["old_bits"],
            "new_bits": best["new_bits"],
            "wga_before": current_wga,
            "wga_after": best["probe_wga"],
            "delta_wga": best["delta_wga"],
            "delta_bits": best["delta_bits"],
            "ratio": best["ratio"],
            "total_budget_bits_after": new_bits,
            "compression_ratio_after": round(new_cr, 4),
            "ud_score_after": best["probe_metrics"].get("unfairness_score"),
            "top1_after": best["probe_metrics"]["top1"],
            "macro_f1_after": best["probe_metrics"].get("macro_f1"),
            "group_accuracies_after": best["probe_metrics"].get("group_accuracies"),
            "candidates_explored": [
                {
                    "layer": r["layer"],
                    "component": r["component"],
                    "old_bits": r["old_bits"],
                    "new_bits": r["new_bits"],
                    "delta_wga": r["delta_wga"],
                    "delta_bits": r["delta_bits"],
                    "ratio": r["ratio"],
                    "probe_wga": r["probe_wga"],
                    "ud_score": r["probe_metrics"].get("unfairness_score"),
                    "top1": r["probe_metrics"]["top1"],
                    "macro_f1": r["probe_metrics"].get("macro_f1"),
                    "group_accuracies": r["probe_metrics"].get("group_accuracies"),
                }
                for r in candidate_results
            ],
        })

        # Advance state
        current_wga  = best["probe_wga"]
        current_bits = new_bits
    else:
        # Loop exhausted all feasible upgrades without break
        history_stopping_reason = "no_candidates"

    return assignment_wts, assignment_acts, history, all_candidates_log, history_stopping_reason


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Greedy fairness-aware mixed-precision quantization search.\n"
            "Optimises worst-group accuracy (WGA) within a memory budget.\n"
            "Imports backend from ../backend/ (Option A — no code duplication)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Skin dataset args ─────────────────────────────────────────────────────
    parser.add_argument("--skin-csv",        required=True,  help="Path to fitzpatrick17k.csv")
    parser.add_argument("--skin-image-dir",  required=True,  help="Path to image directory")
    parser.add_argument("--skin-label-space", choices=["auto", "binary", "nine"], default="nine")
    parser.add_argument("--skin-eval-split",  choices=["val", "test"], default="test")
    parser.add_argument("--skin-batch-size",  type=int, default=32)

    # ── Model args ────────────────────────────────────────────────────────────
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path. Auto-selected from default candidates if omitted.")
    parser.add_argument("--distiller-dir",
                        default="/home/mussa/skin-fairness-ptq/distiller")
    parser.add_argument("--arch", default="resnet18",
                        help="Model architecture (must match the checkpoint). "
                             "E.g. resnet18, resnet32, resnet56, mobilenet_v2, shufflenet_v2_x1_0, vgg11.")

    # ── Eval args ─────────────────────────────────────────────────────────────
    parser.add_argument("--eval-batches", type=int, default=0,
                        help="Batches per eval (0 = full test set).")
    parser.add_argument("--seed", type=int, default=42)

    # ── Bit options ───────────────────────────────────────────────────────────
    parser.add_argument("--bit-options-wts",  default="8,7,6,5,4",
                        help="Comma-separated weight bit options (descending). Must include 8 and start-bits-wts.")
    parser.add_argument("--bit-options-acts", default="8,7,6,5,4",
                        help="Comma-separated activation bit options. Must include 8 and start-bits-acts.")

    # ── Greedy algorithm args ─────────────────────────────────────────────────
    parser.add_argument("--start-bits-wts",   type=int,   default=4,
                        help="Starting weight precision for all layers (lowest point).")
    parser.add_argument("--start-bits-acts",  type=int,   default=4,
                        help="Starting activation precision for all layers (lowest point).")
    parser.add_argument("--budget-ratio",     type=float, default=4.0,
                        help=(
                            "Memory budget expressed as a compression ratio vs FP32 weights. "
                            "E.g. 4.0 means the quantized weights may use at most 1/4 of FP32 memory. "
                            "Search stops when next upgrade would exceed this budget."
                        ))
    parser.add_argument("--budget-mode",
                        choices=["weights", "combined_proxy"], default="weights",
                        help=(
                            "'weights' (default): budget tracks weight bits only. "
                            "'combined_proxy': budget tracks weight + activation bits (using weight_params as proxy)."
                        ))
    parser.add_argument("--no-upgrade-acts",  action="store_true", default=False,
                        help="Disable activation precision upgrades (weights-only ablation).")
    parser.add_argument("--plateau-patience", type=int, default=0,
                        help=(
                            "Allow N consecutive iterations with no WGA improvement before stopping. "
                            "0 (default) = stop immediately on first non-improving iteration."
                        ))
    parser.add_argument("--max-ud", type=float, default=None,
                        help=(
                            "Hard UD ceiling. Candidates whose unfairness disparity exceeds this "
                            "value are rejected and not eligible for selection. "
                            "Typical use: set to W8A8_UD + 0.02 per arch."
                        ))

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument("--output-dir", default="",
                        help="Output directory. Auto-generated under results/ if omitted.")

    args = parser.parse_args()

    # ── Checkpoint selection ──────────────────────────────────────────────────
    if args.checkpoint is None:
        candidates = _default_checkpoint_candidates(args.arch, args.skin_label_space)
        # Legacy fallback for the original resnet18 run
        if args.arch == "resnet18":
            candidates += _DEFAULT_CHECKPOINT_CANDIDATES_RESNET18
        args.checkpoint = _pick_existing_checkpoint(candidates)
        if args.checkpoint is None:
            parser.error(
                "No default checkpoint found. Pass --checkpoint explicitly.\n"
                f"Searched: {candidates}"
            )

    # ── Output directory ──────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        results_root = Path(__file__).parent / "results"
        output_dir = results_root / f"run_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir)
    logger.info("Greedy Fairness-Aware Mixed-Precision Search")
    logger.info("Output directory: %s", output_dir)
    logger.info("Checkpoint: %s", args.checkpoint)

    # ── Parse and validate bit options ───────────────────────────────────────
    w_bits = parse_bits(args.bit_options_wts)
    a_bits = parse_bits(args.bit_options_acts)
    upgrade_acts = not args.no_upgrade_acts

    if 8 not in w_bits:
        parser.error("--bit-options-wts must include 8 (needed for baseline PTQ W8/A8)")
    if 8 not in a_bits:
        parser.error("--bit-options-acts must include 8 (needed for baseline PTQ W8/A8)")
    if args.start_bits_wts not in w_bits:
        parser.error(f"--start-bits-wts={args.start_bits_wts} must be in --bit-options-wts ({w_bits})")
    if args.start_bits_acts not in a_bits:
        parser.error(f"--start-bits-acts={args.start_bits_acts} must be in --bit-options-acts ({a_bits})")

    logger.info(
        "Config: budget_ratio=%.1f mode=%s upgrade_acts=%s plateau_patience=%d max_ud=%s",
        args.budget_ratio, args.budget_mode, upgrade_acts, args.plateau_patience,
        f"{args.max_ud:.4f}" if args.max_ud is not None else "None",
    )
    logger.info("Weight bit options: %s (start=%d)", w_bits, args.start_bits_wts)
    logger.info("Activation bit options: %s (start=%d)", a_bits, args.start_bits_acts)

    # ── Build backend ─────────────────────────────────────────────────────────
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
        arch=args.arch,
    )
    cache_path = output_dir / "eval_cache.json"
    backend.load_cache(cache_path)

    # ── Build layer inventory ─────────────────────────────────────────────────
    model = backend.load_model_for_inventory()
    inventory = build_layer_inventory(model)
    (output_dir / "layer_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    layers = [entry["name"] for entry in inventory["layers"]]
    params_by_layer = {entry["name"]: int(entry["weight_params"]) for entry in inventory["layers"]}
    logger.info("Quantizable layers: %d | Total weight params: %d", len(layers), sum(params_by_layer.values()))

    # ── Stage 0: baseline (FP32) ──────────────────────────────────────────────
    logger.info("Stage 0a: evaluating FP32 baseline...")
    baseline = backend.evaluate_baseline()
    (output_dir / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    baseline_wga = compute_wga(baseline)
    logger.info(
        "Baseline FP32 | top1=%.3f | WGA=%.4f | UD=%.4f | macro_f1=%.4f",
        baseline["top1"], baseline_wga,
        baseline.get("unfairness_score", float("nan")),
        baseline.get("macro_f1", float("nan")),
    )

    # ── Stage 0: uniform PTQ W8/A8 ───────────────────────────────────────────
    logger.info("Stage 0b: evaluating uniform PTQ W8/A8...")
    w8a8_assignment = {layer: 8 for layer in layers}
    ptq_default = backend.evaluate_quantized(
        assignment_wts=w8a8_assignment,
        assignment_acts=w8a8_assignment,
    )
    (output_dir / "ptq_default.json").write_text(json.dumps(ptq_default, indent=2) + "\n")
    logger.info(
        "Uniform W8/A8 | top1=%.3f | WGA=%.4f | UD=%.4f | macro_f1=%.4f",
        ptq_default["top1"], compute_wga(ptq_default),
        ptq_default.get("unfairness_score", float("nan")),
        ptq_default.get("macro_f1", float("nan")),
    )

    # ── Stage 0: starting precision (W4A4) ───────────────────────────────────
    logger.info("Stage 0c: evaluating starting precision W%dA%d...", args.start_bits_wts, args.start_bits_acts)
    start_assignment_wts  = {layer: args.start_bits_wts  for layer in layers}
    start_assignment_acts = {layer: args.start_bits_acts for layer in layers}
    ptq_start = backend.evaluate_quantized(
        assignment_wts=start_assignment_wts,
        assignment_acts=start_assignment_acts,
    )
    (output_dir / "ptq_start.json").write_text(json.dumps(ptq_start, indent=2) + "\n")
    ptq_start_wga = compute_wga(ptq_start)
    logger.info(
        "Uniform W%dA%d | top1=%.3f | WGA=%.4f | UD=%.4f | macro_f1=%.4f",
        args.start_bits_wts, args.start_bits_acts,
        ptq_start["top1"], ptq_start_wga,
        ptq_start.get("unfairness_score", float("nan")),
        ptq_start.get("macro_f1", float("nan")),
    )

    # ── Stage 1: greedy fairness search ──────────────────────────────────────
    logger.info("Stage 1: running greedy fairness search...")
    assignment_wts, assignment_acts, history, all_candidates_log, stopping_reason = greedy_fairness_search(
        backend=backend,
        layers=layers,
        params_by_layer=params_by_layer,
        w_bits=w_bits,
        a_bits=a_bits,
        start_bits_wts=args.start_bits_wts,
        start_bits_acts=args.start_bits_acts,
        budget_ratio=args.budget_ratio,
        budget_mode=args.budget_mode,
        upgrade_acts=upgrade_acts,
        plateau_patience=args.plateau_patience,
        logger=logger,
        max_ud=args.max_ud,
    )
    (output_dir / "fairness_search_history.json").write_text(
        json.dumps(history, indent=2) + "\n"
    )
    (output_dir / "all_candidates_log.json").write_text(
        json.dumps(all_candidates_log, indent=2) + "\n"
    )
    logger.info("Greedy search done: %d iterations, stopping_reason=%s", len(history), stopping_reason)

    # ── Stage 2: final eval of resulting assignment ───────────────────────────
    logger.info("Stage 2: final eval of search result...")
    final_metrics = backend.evaluate_quantized(
        assignment_wts=assignment_wts,
        assignment_acts=assignment_acts,
    )
    (output_dir / "assignment_weights.json").write_text(json.dumps(assignment_wts, indent=2) + "\n")
    (output_dir / "assignment_activations.json").write_text(json.dumps(assignment_acts, indent=2) + "\n")

    # ── Compute compression metrics ───────────────────────────────────────────
    fp32_weight_bits = sum(params_by_layer.values()) * 32
    weight_bits = theoretical_size_bits(assignment_wts, params_by_layer)
    activation_bits_proxy = theoretical_size_bits(assignment_acts, params_by_layer)
    combined_bits_proxy = weight_bits + activation_bits_proxy
    compression_ratio_weights = fp32_weight_bits / weight_bits if weight_bits else 0.0

    final_wga = compute_wga(final_metrics)

    # ── Build final.json ──────────────────────────────────────────────────────
    final = {
        # Standard fields (same keys as backend runs for easy comparison)
        "baseline_top1":                  baseline["top1"],
        "ptq_default_top1":               ptq_default["top1"],
        "ptq_start_top1":                 ptq_start["top1"],
        "final_top1":                     final_metrics["top1"],
        "drop_vs_baseline":               baseline["top1"] - final_metrics["top1"],
        "arch":                           args.arch,
        "task":                           "skin_fitzpatrick",
        "skin_label_space":               args.skin_label_space,
        "checkpoint":                     str(Path(args.checkpoint).resolve()),
        "seed":                           args.seed,
        "eval_batches":                   args.eval_batches,
        "weight_bit_options":             w_bits,
        "activation_bit_options":         a_bits,
        "weights_theoretical_bits":       weight_bits,
        "fp32_weights_bits":              fp32_weight_bits,
        "compression_ratio_vs_fp32_weights": compression_ratio_weights,
        "activations_theoretical_bits_proxy": activation_bits_proxy,
        "combined_theoretical_bits_proxy":    combined_bits_proxy,
        "weights_bit_distribution":       bit_distribution(assignment_wts),
        "activations_bit_distribution":   bit_distribution(assignment_acts),
        "num_evals_executed":             backend.eval_count,
        # WGA fields
        "baseline_wga":                   baseline_wga,
        "ptq_start_wga":                  ptq_start_wga,
        "ptq_default_wga":                compute_wga(ptq_default),
        "final_wga":                      final_wga,
        "wga_improvement_vs_baseline":    final_wga - baseline_wga,
        "wga_improvement_vs_ptq_start":   final_wga - ptq_start_wga,
        # Algorithm config
        "search_algorithm":               "greedy_fairness",
        "budget_ratio":                   args.budget_ratio,
        "budget_mode":                    args.budget_mode,
        "start_bits_wts":                 args.start_bits_wts,
        "start_bits_acts":                args.start_bits_acts,
        "upgrade_acts":                   upgrade_acts,
        "plateau_patience":               args.plateau_patience,
        "total_iterations":               len(history),
        "stopping_reason":                stopping_reason,
        "max_ud_constraint":              args.max_ud,
    }

    # Mirror baseline/final skin metric fields (same pattern as original search.py)
    for metric_key in [
        "accuracy", "macro_f1", "weighted_f1",
        "sensitivity", "specificity", "precision", "f1",
        "tp", "tn", "fp", "fn",
        "n_samples", "num_classes", "label_space", "class_names", "eval_split",
        "unfairness_score", "group_accuracies",
    ]:
        if metric_key in baseline:
            final[f"baseline_{metric_key}"] = baseline[metric_key]
        if metric_key in final_metrics:
            final[f"final_{metric_key}"] = final_metrics[metric_key]

    (output_dir / "final.json").write_text(json.dumps(final, indent=2) + "\n")
    backend.save_cache(cache_path)

    # ── Console summary ───────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info("%-30s  top1=%6.2f  WGA=%.4f  UD=%.4f  macro_f1=%.4f",
                "Baseline (FP32)",
                baseline["top1"], baseline_wga,
                baseline.get("unfairness_score", float("nan")),
                baseline.get("macro_f1", float("nan")))
    logger.info("%-30s  top1=%6.2f  WGA=%.4f  UD=%.4f  macro_f1=%.4f",
                "Uniform W8/A8 (naive PTQ)",
                ptq_default["top1"], compute_wga(ptq_default),
                ptq_default.get("unfairness_score", float("nan")),
                ptq_default.get("macro_f1", float("nan")))
    logger.info("%-30s  top1=%6.2f  WGA=%.4f  UD=%.4f  macro_f1=%.4f",
                f"Uniform W{args.start_bits_wts}A{args.start_bits_acts} (start)",
                ptq_start["top1"], ptq_start_wga,
                ptq_start.get("unfairness_score", float("nan")),
                ptq_start.get("macro_f1", float("nan")))
    logger.info("%-30s  top1=%6.2f  WGA=%.4f  UD=%.4f  macro_f1=%.4f  compression=%.2fx  iters=%d",
                "Greedy Fairness (ours)",
                final_metrics["top1"], final_wga,
                final_metrics.get("unfairness_score", float("nan")),
                final_metrics.get("macro_f1", float("nan")),
                compression_ratio_weights, len(history))
    logger.info("Stopping reason: %s | Total evals: %d", stopping_reason, backend.eval_count)
    logger.info("Output: %s", output_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
