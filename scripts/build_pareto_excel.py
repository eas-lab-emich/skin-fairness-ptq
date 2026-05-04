"""
build_pareto_excel.py
=====================
Produces pareto_full.xlsx with two views of the greedy search data so you
can decide with your professor which is right for the Pareto front:

  Sheet "Baselines"          — FP32 / W8A8 / W6A6 for all 6 archs
  Sheet "{arch} Path"        — accepted model states as the greedy walked
                               from W6A6 toward W8A8 (row 0 = W6A6 start,
                               row N = after Nth accepted upgrade).
                               Each row is a REAL model configuration.
  Sheet "{arch} Probes"      — every layer-level candidate the greedy
                               evaluated (accepted OR rejected).
                               Each row is a single-layer "what if" probe,
                               NOT a complete model state.

Run from project root:
  python3 scripts/build_pareto_excel.py
"""

import json
from pathlib import Path
import pandas as pd

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "search" / "results"
OUT_PATH    = RESULTS_DIR / "pareto_full.xlsx"

ARCHS = ["vgg11", "resnet18", "resnet34", "resnet50", "mobilenet_v2", "shufflenet_v2_x1_0"]

ARCH_LABEL = {
    "vgg11":              "vgg11",
    "resnet18":           "resnet18",
    "resnet34":           "resnet34",
    "resnet50":           "resnet50",
    "mobilenet_v2":       "mobilenet_v2",
    "shufflenet_v2_x1_0": "shufflenet",
}

# ── Hardcoded baselines ───────────────────────────────────────────────────────

FP32 = {
    "vgg11":              {"top1": 76.64, "wga": None,  "ud": 0.210, "macro_f1": 0.581, "avg_bits": 32.0},
    "resnet18":           {"top1": 78.23, "wga": None,  "ud": 0.237, "macro_f1": 0.619, "avg_bits": 32.0},
    "resnet34":           {"top1": 78.64, "wga": None,  "ud": 0.181, "macro_f1": 0.638, "avg_bits": 32.0},
    "resnet50":           {"top1": 78.31, "wga": None,  "ud": 0.151, "macro_f1": 0.623, "avg_bits": 32.0},
    "mobilenet_v2":       {"top1": 77.02, "wga": None,  "ud": 0.158, "macro_f1": 0.591, "avg_bits": 32.0},
    "shufflenet_v2_x1_0": {"top1": 74.52, "wga": None,  "ud": 0.234, "macro_f1": 0.574, "avg_bits": 32.0},
}

W8A8 = {
    "vgg11":              {"top1": 76.27, "wga": 0.696, "ud": 0.125, "macro_f1": 0.571, "avg_bits": 8.0},
    "resnet18":           {"top1": 77.56, "wga": 0.722, "ud": 0.107, "macro_f1": 0.610, "avg_bits": 8.0},
    "resnet34":           {"top1": 78.39, "wga": 0.751, "ud": 0.074, "macro_f1": 0.633, "avg_bits": 8.0},
    "resnet50":           {"top1": 78.14, "wga": 0.735, "ud": 0.101, "macro_f1": 0.624, "avg_bits": 8.0},
    "mobilenet_v2":       {"top1": 77.27, "wga": 0.728, "ud": 0.101, "macro_f1": 0.592, "avg_bits": 8.0},
    "shufflenet_v2_x1_0": {"top1": 72.36, "wga": 0.687, "ud": 0.095, "macro_f1": 0.543, "avg_bits": 8.0},
}

W6A6 = {
    "vgg11":              {"top1": 70.40, "wga": 0.662, "ud": 0.098, "macro_f1": 0.499, "avg_bits": 6.0},
    "resnet18":           {"top1": 73.69, "wga": 0.701, "ud": 0.110, "macro_f1": 0.537, "avg_bits": 6.0},
    "resnet34":           {"top1": 72.15, "wga": 0.662, "ud": 0.134, "macro_f1": 0.531, "avg_bits": 6.0},
    "resnet50":           {"top1": 69.44, "wga": 0.662, "ud": 0.074, "macro_f1": 0.481, "avg_bits": 6.0},
    "mobilenet_v2":       {"top1": 61.28, "wga": 0.575, "ud": 0.087, "macro_f1": 0.221, "avg_bits": 6.0},
    "shufflenet_v2_x1_0": {"top1": 58.70, "wga": 0.540, "ud": 0.136, "macro_f1": 0.201, "avg_bits": 6.0},
}


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def wga_from_groups(group_accs):
    if not group_accs:
        return None
    return min(float(v) for v in group_accs.values())


def ud_from_groups(group_accs):
    if not group_accs:
        return None
    vals = [float(v) for v in group_accs.values()]
    return max(vals) - min(vals)


# ── Baselines sheet ───────────────────────────────────────────────────────────

def build_baselines():
    rows = []
    for arch in ARCHS:
        for method, table in [("FP32", FP32), ("W8A8 (uniform)", W8A8), ("W6A6 (uniform)", W6A6)]:
            m = table[arch]
            rows.append({
                "arch":      arch,
                "method":    method,
                "top1":      m["top1"],
                "wga":       m.get("wga"),
                "ud":        m["ud"],
                "macro_f1":  m["macro_f1"],
                "avg_bits":  m["avg_bits"],
            })
    return pd.DataFrame(rows)


# ── Path sheet (accepted model states) ───────────────────────────────────────

def build_path(arch):
    """
    One row per accepted greedy step, starting from W6A6 (iteration 0).
    Each row = a real, complete model configuration.
    """
    result_dir = RESULTS_DIR / f"greedy_ud_constrained_{arch}_s42"
    ptq_start  = load(result_dir / "ptq_start.json")
    history    = load(result_dir / "fairness_search_history.json") or []

    rows = []

    # Row 0: W6A6 starting state
    if ptq_start:
        ga = ptq_start.get("group_accuracies", {})
        rows.append({
            "iteration":         0,
            "layer_upgraded":    "W6A6 start",
            "component":         None,
            "old_bits":          None,
            "new_bits":          None,
            "top1":              ptq_start.get("top1"),
            "wga":               wga_from_groups(ga),
            "ud":                ptq_start.get("unfairness_score") or ud_from_groups(ga),
            "macro_f1":          ptq_start.get("macro_f1"),
            "compression_ratio": None,
            "avg_wt_bits":       6.0,
        })

    # Accepted steps
    for step in history:
        ga = step.get("group_accuracies_after", {})
        rows.append({
            "iteration":         step.get("iteration"),
            "layer_upgraded":    step.get("layer"),
            "component":         step.get("component"),
            "old_bits":          step.get("old_bits"),
            "new_bits":          step.get("new_bits"),
            "top1":              step.get("top1_after"),
            "wga":               step.get("wga_after") or wga_from_groups(ga),
            "ud":                step.get("ud_score_after") or ud_from_groups(ga),
            "macro_f1":          step.get("macro_f1_after"),
            "compression_ratio": step.get("compression_ratio_after"),
            "avg_wt_bits":       None,
        })

    return pd.DataFrame(rows)


# ── Probes sheet (every evaluated candidate) ──────────────────────────────────

def build_probes(arch):
    """
    One row per layer-level probe evaluated during the search.
    These are NOT complete model states — each is a single-layer "what if"
    tested against the current model at that iteration.
    """
    cand_path = RESULTS_DIR / f"greedy_ud_constrained_{arch}_s42" / "all_candidates_log.json"
    candidates = load(cand_path)
    if not candidates:
        return pd.DataFrame()

    rows = []
    for c in candidates:
        rows.append({
            "iteration":           c.get("iteration"),
            "layer":               c.get("layer"),
            "component":           c.get("component"),
            "old_bits":            c.get("old_bits"),
            "new_bits":            c.get("new_bits"),
            "top1":                c.get("top1"),
            "wga":                 c.get("probe_wga"),
            "ud":                  c.get("ud_score"),
            "macro_f1":            c.get("macro_f1"),
            "delta_wga":           c.get("delta_wga"),
            "rejected_ud_ceiling": c.get("rejected_ud_ceiling", False),
        })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:

        # Baselines
        build_baselines().to_excel(writer, sheet_name="Baselines", index=False)
        print("  Baselines sheet written")

        for arch in ARCHS:
            label = ARCH_LABEL[arch]

            # Path sheet
            path_df = build_path(arch)
            sheet   = f"{label} Path"[:31]
            path_df.to_excel(writer, sheet_name=sheet, index=False)
            print(f"  {sheet}: {len(path_df)} rows  (row 0 = W6A6 start, rest = accepted steps)")

            # Probes sheet
            probes_df = build_probes(arch)
            sheet     = f"{label} Probes"[:31]
            probes_df.to_excel(writer, sheet_name=sheet, index=False)
            print(f"  {sheet}: {len(probes_df)} rows")

    print(f"\nSaved: {OUT_PATH}")
    print(f"Sheets: Baselines + {len(ARCHS)} Path + {len(ARCHS)} Probes = {1 + 2*len(ARCHS)} total")


if __name__ == "__main__":
    main()
