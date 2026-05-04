"""
run_uniform_sweep.py
====================
Evaluate uniform quantization baselines (W8A8, W7A7, W6A6, W5A5, W4A4) for a
given model checkpoint, producing one JSON result file per configuration.

Usage example:
  python run_uniform_sweep.py \
    --checkpoint /path/to/fitz17k_nine_mobilenet_v2_best.pth \
    --arch mobilenet_v2 \
    --skin-csv /home/mussa/skin_fairness_project/data/fitzpatrick17k.csv \
    --skin-image-dir /home/mussa/fitzpatrick17k_data/data/finalfitz17k \
    --output-dir results/uniform_sweep_mobilenet_v2
"""

import argparse
import json
import sys
import os
from pathlib import Path

_BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
sys.path.insert(0, _BACKEND_DIR)

from skin_distiller_backend import SkinDistillerBackend  # noqa: E402
from layer_inventory import build_layer_inventory         # noqa: E402

# Default bit configurations to sweep
DEFAULT_CONFIGS = [
    (8, 8),
    (7, 7),
    (6, 6),
    (5, 5),
    (4, 4),
]


def run_sweep(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Discover quantizable layers
    model = backend.load_model_for_inventory()
    inventory = build_layer_inventory(model)
    (output_dir / "layer_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    layers = [entry["name"] for entry in inventory["layers"]]
    print(f"Quantizable layers: {len(layers)}")

    # FP32 baseline
    print("Evaluating FP32 baseline...")
    baseline = backend.evaluate_baseline()
    (output_dir / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"  FP32  top1={baseline['top1']:.2f}  WGA={min(baseline['group_accuracies'].values()):.4f}  macro_f1={baseline['macro_f1']:.4f}")

    # Parse which configs to sweep
    if args.configs:
        configs = []
        for tok in args.configs.split(","):
            tok = tok.strip()
            if "A" in tok.upper():
                w, a = tok.upper().split("A")
                configs.append((int(w.lstrip("W")), int(a)))
            else:
                b = int(tok)
                configs.append((b, b))
    else:
        configs = DEFAULT_CONFIGS

    results = {"arch": args.arch, "checkpoint": args.checkpoint, "baseline": baseline, "configs": {}}

    for w_bits, a_bits in configs:
        tag = f"W{w_bits}A{a_bits}"
        print(f"Evaluating uniform {tag}...")
        assignment = {layer: w_bits for layer in layers}
        assignment_acts = {layer: a_bits for layer in layers}
        metrics = backend.evaluate_quantized(
            assignment_wts=assignment,
            assignment_acts=assignment_acts,
        )
        (output_dir / f"uniform_{tag}.json").write_text(json.dumps(metrics, indent=2) + "\n")
        wga = min(metrics["group_accuracies"].values())
        ud = max(metrics["group_accuracies"].values()) - min(metrics["group_accuracies"].values())
        print(f"  {tag}  top1={metrics['top1']:.2f}  WGA={wga:.4f}  UD={ud:.4f}  macro_f1={metrics['macro_f1']:.4f}")
        results["configs"][tag] = {
            "w_bits": w_bits,
            "a_bits": a_bits,
            "top1": metrics["top1"],
            "wga": wga,
            "ud": ud,
            "macro_f1": metrics["macro_f1"],
            "group_accuracies": metrics["group_accuracies"],
        }

    # Summary
    (output_dir / "sweep_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nSweep complete. Results in: {output_dir}")
    print(f"Total evals executed: {backend.eval_count}")


def parse_args():
    parser = argparse.ArgumentParser(description="Uniform quantization sweep (W8A8 → W4A4) for a given checkpoint.")
    parser.add_argument("--arch", required=True,
                        help="Model architecture (e.g. resnet18, mobilenet_v2).")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint (.pth).")
    parser.add_argument("--skin-csv",
                        default="/home/mussa/skin_fairness_project/data/fitzpatrick17k.csv")
    parser.add_argument("--skin-image-dir",
                        default="/home/mussa/fitzpatrick17k_data/data/finalfitz17k")
    parser.add_argument("--skin-label-space", choices=["auto", "binary", "nine"], default="nine")
    parser.add_argument("--skin-eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--skin-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=0,
                        help="Batches per eval (0 = full test set).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="",
                        help="Output directory. Auto-generated if omitted.")
    parser.add_argument("--distiller-dir",
                        default="/home/mussa/skin-fairness-ptq/distiller")
    parser.add_argument("--configs", default="",
                        help="Comma-separated configs to evaluate, e.g. 'W8A8,W7A7,W6A6,W5A5,W4A4'. "
                             "Defaults to W8A8 through W4A4.")
    args = parser.parse_args()

    if not args.output_dir:
        results_root = Path(__file__).parent / "results"
        args.output_dir = str(results_root / f"uniform_sweep_{args.arch}")

    return args


if __name__ == "__main__":
    run_sweep(parse_args())
