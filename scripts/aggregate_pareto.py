"""
aggregate_pareto.py
===================
Post-run consolidation script. Reads results from UD-constrained greedy runs
and produces two clean files for paper analysis:

  1. pareto_candidates.csv  — every evaluated candidate from all 6 models
                               (for Pareto front plots)
  2. final_comparison.csv   — one row per (arch, method) summary table

Run after all greedy_ud_constrained_<arch>_s42/ runs complete:
  python3 scripts/aggregate_pareto.py
"""

import json
from pathlib import Path
import csv

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "search" / "results"
OUT_DIR     = ROOT / "search" / "results"

ARCHS = ["vgg11", "resnet18", "resnet34", "resnet50", "mobilenet_v2", "shufflenet_v2_x1_0"]

# ── W8A8 UD baselines (from uniform sweep) ────────────────────────────────────
W8A8_UD = {
    "vgg11":               0.125,
    "resnet18":            0.107,
    "resnet34":            0.074,
    "resnet50":            0.101,
    "mobilenet_v2":        0.101,
    "shufflenet_v2_x1_0":  0.095,
}

# ── W8A8 metrics (from uniform sweep) ─────────────────────────────────────────
W8A8_METRICS = {
    "vgg11":               {"top1": 76.27, "wga": 0.696, "ud": 0.125, "macro_f1": 0.571},
    "resnet18":            {"top1": 77.56, "wga": 0.722, "ud": 0.107, "macro_f1": 0.610},
    "resnet34":            {"top1": 78.39, "wga": 0.751, "ud": 0.074, "macro_f1": 0.633},
    "resnet50":            {"top1": 78.14, "wga": 0.735, "ud": 0.101, "macro_f1": 0.624},
    "mobilenet_v2":        {"top1": 77.27, "wga": 0.728, "ud": 0.101, "macro_f1": 0.592},
    "shufflenet_v2_x1_0":  {"top1": 72.36, "wga": 0.687, "ud": 0.095, "macro_f1": 0.543},
}

# ── W6A6 uniform metrics (from uniform sweep) ─────────────────────────────────
W6A6_METRICS = {
    "vgg11":               {"top1": 70.40, "wga": 0.662, "ud": 0.098, "macro_f1": 0.499},
    "resnet18":            {"top1": 73.69, "wga": 0.701, "ud": 0.110, "macro_f1": 0.537},
    "resnet34":            {"top1": 72.15, "wga": 0.662, "ud": 0.134, "macro_f1": 0.531},
    "resnet50":            {"top1": 69.44, "wga": 0.662, "ud": 0.074, "macro_f1": 0.481},
    "mobilenet_v2":        {"top1": 61.28, "wga": 0.575, "ud": 0.087, "macro_f1": 0.221},
    "shufflenet_v2_x1_0":  {"top1": 58.70, "wga": 0.540, "ud": 0.136, "macro_f1": 0.201},
}

# ── FP32 metrics (from baseline) ──────────────────────────────────────────────
FP32_METRICS = {
    "vgg11":               {"top1": 76.64, "wga": None,  "ud": 0.210, "macro_f1": 0.581},
    "resnet18":            {"top1": 78.23, "wga": None,  "ud": 0.237, "macro_f1": 0.619},
    "resnet34":            {"top1": 78.64, "wga": None,  "ud": 0.181, "macro_f1": 0.638},
    "resnet50":            {"top1": 78.31, "wga": None,  "ud": 0.151, "macro_f1": 0.623},
    "mobilenet_v2":        {"top1": 77.02, "wga": None,  "ud": 0.158, "macro_f1": 0.591},
    "shufflenet_v2_x1_0":  {"top1": 74.52, "wga": None,  "ud": 0.234, "macro_f1": 0.574},
}

# ── FP32 model sizes (MB) ─────────────────────────────────────────────────────
FP32_MB = {
    "vgg11": 515.2, "resnet18": 44.7, "resnet34": 85.1,
    "resnet50": 93.9, "mobilenet_v2": 8.8, "shufflenet_v2_x1_0": 5.0,
}

# ── Unconstrained greedy final results (from previous runs) ───────────────────
GREEDY_UNCONSTRAINED = {
    "vgg11":               {"top1": 73.86, "wga": 0.709, "ud": 0.176, "macro_f1": 0.526, "avg_wt_bits": 7.88},
    "resnet18":            {"top1": 77.56, "wga": 0.722, "ud": 0.229, "macro_f1": 0.610, "avg_wt_bits": 8.00},
    "resnet34":            {"top1": 77.94, "wga": 0.746, "ud": 0.182, "macro_f1": 0.623, "avg_wt_bits": 7.83},
    "resnet50":            {"top1": 78.56, "wga": 0.764, "ud": 0.133, "macro_f1": 0.627, "avg_wt_bits": 7.99},
    "mobilenet_v2":        {"top1": 70.19, "wga": 0.683, "ud": 0.196, "macro_f1": 0.308, "avg_wt_bits": 6.72},
    "shufflenet_v2_x1_0":  {"top1": None,  "wga": None,  "ud": None,  "macro_f1": None,  "avg_wt_bits": 8.00},
}


def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def compute_wga(metrics):
    ga = metrics.get("group_accuracies") if metrics else None
    if not ga:
        return None
    return min(float(v) for v in ga.values())


def param_weighted_avg_bits(assignment, inventory):
    layers = {l["name"]: l["weight_params"] for l in inventory["layers"]}
    total_params = inventory["total_weight_params"]
    weighted = sum(layers.get(n, 0) * b for n, b in assignment.items())
    return weighted / total_params if total_params else 0


def model_size_mb(assignment, inventory):
    layers = {l["name"]: l["weight_params"] for l in inventory["layers"]}
    weighted_bits = sum(layers.get(n, 0) * b for n, b in assignment.items())
    return weighted_bits / 8 / 1e6


PARETO_FIELDS = [
    "iteration", "layer", "component", "old_bits", "new_bits",
    "delta_wga", "delta_bits", "ratio", "probe_wga",
    "ud_score", "top1", "macro_f1", "rejected_ud_ceiling",
]


def build_pareto_csv():
    import pandas as pd

    all_rows = []
    per_arch = {}

    for arch in ARCHS:
        cand_path = RESULTS_DIR / f"greedy_ud_constrained_{arch}_s42" / "all_candidates_log.json"
        if not cand_path.exists():
            print(f"  [MISSING] {cand_path}")
            continue
        candidates = json.loads(cand_path.read_text())
        rows = []
        for c in candidates:
            row = {"arch": arch}
            for f in PARETO_FIELDS:
                row[f] = c.get(f, False if f == "rejected_ud_ceiling" else None)
            rows.append(row)
        per_arch[arch] = rows
        all_rows.extend(rows)
        print(f"  {arch}: {len(candidates)} candidates loaded")

    if not all_rows:
        print("No pareto data found. Run the constrained greedy first.")
        return

    # Flat CSV (one file, all archs)
    csv_path = OUT_DIR / "pareto_candidates.csv"
    fieldnames = list(all_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nPareto CSV saved: {csv_path}  ({len(all_rows)} rows)")

    # Excel with one sheet per arch
    xlsx_path = OUT_DIR / "pareto_candidates.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for arch, rows in per_arch.items():
            df = pd.DataFrame(rows, columns=["arch"] + PARETO_FIELDS)
            sheet_name = arch[:31]  # Excel sheet name limit
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Pareto Excel saved: {xlsx_path}  ({len(per_arch)} sheets)")


def build_comparison_csv():
    rows = []

    for arch in ARCHS:
        fp32_mb = FP32_MB[arch]

        # FP32
        fp32 = FP32_METRICS[arch]
        rows.append({
            "arch": arch, "method": "FP32",
            "top1": fp32["top1"], "wga": fp32.get("wga"), "ud": fp32["ud"],
            "macro_f1": fp32["macro_f1"], "avg_wt_bits": 32.0,
            "model_size_mb": fp32_mb, "compression_vs_fp32": 1.0,
            "stopping_reason": None, "total_iterations": None,
        })

        # W8A8
        w8 = W8A8_METRICS[arch]
        rows.append({
            "arch": arch, "method": "W8A8 (uniform)",
            "top1": w8["top1"], "wga": w8["wga"], "ud": w8["ud"],
            "macro_f1": w8["macro_f1"], "avg_wt_bits": 8.0,
            "model_size_mb": round(fp32_mb / 4, 2), "compression_vs_fp32": 4.0,
            "stopping_reason": None, "total_iterations": None,
        })

        # W6A6
        w6 = W6A6_METRICS[arch]
        rows.append({
            "arch": arch, "method": "W6A6 (uniform)",
            "top1": w6["top1"], "wga": w6["wga"], "ud": w6["ud"],
            "macro_f1": w6["macro_f1"], "avg_wt_bits": 6.0,
            "model_size_mb": round(fp32_mb * 6 / 32, 2), "compression_vs_fp32": round(32/6, 2),
            "stopping_reason": None, "total_iterations": None,
        })

        # Greedy unconstrained
        gu = GREEDY_UNCONSTRAINED.get(arch, {})
        gu_size = round(fp32_mb * gu["avg_wt_bits"] / 32, 2) if gu.get("avg_wt_bits") else None
        rows.append({
            "arch": arch, "method": "Greedy W6A6 (unconstrained)",
            "top1": gu.get("top1"), "wga": gu.get("wga"), "ud": gu.get("ud"),
            "macro_f1": gu.get("macro_f1"), "avg_wt_bits": gu.get("avg_wt_bits"),
            "model_size_mb": gu_size,
            "compression_vs_fp32": round(32 / gu["avg_wt_bits"], 2) if gu.get("avg_wt_bits") else None,
            "stopping_reason": None, "total_iterations": None,
        })

        # Greedy UD-constrained (from actual run results)
        constrained_dir = RESULTS_DIR / f"greedy_ud_constrained_{arch}_s42"
        final = load_json(constrained_dir / "final.json")
        inv   = load_json(constrained_dir / "layer_inventory.json")
        wt    = load_json(constrained_dir / "assignment_weights.json")

        if final and inv and wt:
            avg_bits = param_weighted_avg_bits(wt, inv)
            size_mb  = model_size_mb(wt, inv)
            rows.append({
                "arch": arch, "method": "Greedy W6A6 (UD-constrained)",
                "top1": final.get("final_top1"),
                "wga": final.get("final_wga"),
                "ud": final.get("final_unfairness_score"),
                "macro_f1": final.get("final_macro_f1"),
                "avg_wt_bits": round(avg_bits, 3),
                "model_size_mb": round(size_mb, 2),
                "compression_vs_fp32": round(32 / avg_bits, 2) if avg_bits else None,
                "stopping_reason": final.get("stopping_reason"),
                "total_iterations": final.get("total_iterations"),
            })
        else:
            rows.append({
                "arch": arch, "method": "Greedy W6A6 (UD-constrained)",
                "top1": None, "wga": None, "ud": None, "macro_f1": None,
                "avg_wt_bits": None, "model_size_mb": None, "compression_vs_fp32": None,
                "stopping_reason": "NOT_RUN", "total_iterations": None,
            })

    out_path = OUT_DIR / "final_comparison.csv"
    fieldnames = ["arch", "method", "top1", "wga", "ud", "macro_f1",
                  "avg_wt_bits", "model_size_mb", "compression_vs_fp32",
                  "stopping_reason", "total_iterations"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Comparison CSV saved: {out_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    print("=== Building Pareto candidates CSV ===")
    build_pareto_csv()
    print("\n=== Building final comparison CSV ===")
    build_comparison_csv()
    print("\nDone.")
