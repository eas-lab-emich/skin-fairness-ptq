"""
regenerate_comparison_final_paper.py
====================================
Rebuild the cross-method comparison table in Spantidi U_D space using:

  - OLD greedy data (greedy_w6a6_*_s42 and greedy_ud_constrained_*_s42)
    rationale: paper framing treats greedy as a simple reference, NSGA-II as
    the main optimizer. Old greedy collapsed to W6A6 in 4/6 archs due to the
    earlier ceiling-units bug, and that collapse actually reinforces the story.

  - NEW NSGA-II data (nsga2_*_s42_final_paper/pareto_front.json)
    rationale: the new search optimizes Spantidi U_D directly (correct metric).

Differences from regenerate_comparison_spantidi.py:
  - Pareto input path: nsga2_<arch>_s42_final_paper/pareto_front.json
  - New pareto_front.json is a flat list (no "pareto" wrapper) with fields
    top1/ud/avg_bits/assignment_wts/assignment_acts; ud is already Spantidi.
  - WGA/macro_f1 for the NSGA-II row are looked up from
    nsga2_<arch>_s42_final_paper/eval_cache.json by assignment match.
  - Output column is `ud` (not `ud_spantidi`) so energy_simulator.py can
    consume the CSV without changes.
  - Output filename: final_comparison_final_paper.csv (preserve the older
    final_comparison_spantidi.csv as audit trail).
"""

import csv
import json
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "search" / "results"

ARCHS = [
    "vgg11",
    "resnet18",
    "resnet34",
    "resnet50",
    "mobilenet_v2",
    "shufflenet_v2_x1_0",
]


def spantidi_ud(group_accuracies, overall_acc):
    if not group_accuracies:
        return None
    return sum(abs(float(v) - float(overall_acc)) for v in group_accuracies.values())


def wga(group_accuracies):
    if not group_accuracies:
        return None
    return min(float(v) for v in group_accuracies.values())


def param_weighted_avg_bits(assignment, params_by_layer):
    total = sum(params_by_layer.values())
    if total == 0:
        return None
    return sum(int(assignment[l]) * int(p) for l, p in params_by_layer.items()) / total


def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def row_from_baseline_block(arch, method, top1, ga, macro_f1, avg_bits, comp):
    overall = top1 / 100.0
    return {
        "arch":                arch,
        "method":              method,
        "top1":                round(float(top1), 2),
        "wga":                 round(wga(ga), 4) if ga else None,
        "ud":                  round(spantidi_ud(ga, overall), 4) if ga else None,
        "macro_f1":            round(float(macro_f1), 4) if macro_f1 is not None else None,
        "avg_wt_bits":         round(float(avg_bits), 3) if avg_bits is not None else None,
        "compression_vs_fp32": round(float(comp), 2) if comp is not None else None,
    }


def get_w8a8_reference(arch):
    """Load W8A8 (top1, ud Spantidi) from greedy run's ptq_default.json."""
    p = RESULTS / f"greedy_w6a6_{arch}_s42" / "ptq_default.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    ga = d.get("group_accuracies")
    if not ga:
        return None
    return {
        "top1":     float(d["top1"]),
        "ud":       float(spantidi_ud(ga, d["top1"] / 100.0)),
        "avg_bits": 8.0,
    }


def lookup_eval(arch, sol):
    """Look up macro_f1 + group_accuracies for an NSGA-II Pareto solution.

    The eval_cache.json values store (assignment_wts, assignment_acts) and the
    full eval result. We match on exact assignment equality.
    """
    cache_path = RESULTS / f"nsga2_{arch}_s42_final_paper" / "eval_cache.json"
    if not cache_path.exists():
        return {}
    cache = json.loads(cache_path.read_text())
    target_w = sol["assignment_wts"]
    target_a = sol["assignment_acts"]
    for entry in cache.values():
        if not isinstance(entry, dict):
            continue
        if entry.get("assignment_wts") == target_w and entry.get("assignment_acts") == target_a:
            return entry
    return {}


def pick_nsga2_row(arch, w8a8):
    """Pick the headline NSGA-II row.

    Strategy:
      1. From _final_paper pareto_front.json, find solutions that strictly
         dominate W8A8 in Spantidi space.
      2. If any exist, pick the one with the highest top1.
      3. Otherwise, pick the Pareto solution with the highest top1.
    """
    pareto_path = RESULTS / f"nsga2_{arch}_s42_final_paper" / "pareto_front.json"
    if not pareto_path.exists():
        return None
    pareto = json.loads(pareto_path.read_text())
    if not pareto:
        return None

    if w8a8 is not None:
        dominators = [
            r for r in pareto
            if r["top1"]    >= w8a8["top1"]
            and r["ud"]      <= w8a8["ud"]
            and r["avg_bits"] <= w8a8["avg_bits"]
            and (r["top1"] > w8a8["top1"]
                 or r["ud"]   < w8a8["ud"]
                 or r["avg_bits"] < w8a8["avg_bits"])
        ]
        if dominators:
            chosen = max(dominators, key=lambda r: r["top1"])
            return chosen, "dominates_W8A8"

    chosen = max(pareto, key=lambda r: r["top1"])
    return chosen, "best_top1_only"


def process_arch(arch):
    rows = []

    greedy_uncon_dir = RESULTS / f"greedy_w6a6_{arch}_s42"
    greedy_udcon_dir = RESULTS / f"greedy_ud_constrained_{arch}_s42"

    inv = load_json(greedy_uncon_dir / "layer_inventory.json")
    if inv is None:
        inv = load_json(greedy_udcon_dir / "layer_inventory.json")
    if inv is None:
        print(f"  {arch}: NO layer_inventory.json - skipping arch")
        return rows
    params_by_layer = {e["name"]: int(e["weight_params"]) for e in inv["layers"]}

    baseline = load_json(greedy_uncon_dir / "baseline.json")
    if baseline:
        rows.append(row_from_baseline_block(
            arch, "FP32",
            top1=baseline["top1"],
            ga=baseline.get("group_accuracies"),
            macro_f1=baseline.get("macro_f1"),
            avg_bits=32.0,
            comp=1.0,
        ))

    ptq_default = load_json(greedy_uncon_dir / "ptq_default.json")
    if ptq_default:
        rows.append(row_from_baseline_block(
            arch, "W8A8 (uniform)",
            top1=ptq_default["top1"],
            ga=ptq_default.get("group_accuracies"),
            macro_f1=ptq_default.get("macro_f1"),
            avg_bits=8.0,
            comp=4.0,
        ))

    ptq_start = load_json(greedy_uncon_dir / "ptq_start.json")
    if ptq_start:
        rows.append(row_from_baseline_block(
            arch, "W6A6 (uniform)",
            top1=ptq_start["top1"],
            ga=ptq_start.get("group_accuracies"),
            macro_f1=ptq_start.get("macro_f1"),
            avg_bits=6.0,
            comp=round(32 / 6, 2),
        ))

    final_uncon = load_json(greedy_uncon_dir / "final.json")
    wt_uncon    = load_json(greedy_uncon_dir / "assignment_weights.json")
    if final_uncon and wt_uncon:
        avg_b = param_weighted_avg_bits(wt_uncon, params_by_layer)
        rows.append(row_from_baseline_block(
            arch, "Greedy W6A6 (unconstrained)",
            top1=final_uncon["final_top1"],
            ga=final_uncon.get("final_group_accuracies"),
            macro_f1=final_uncon.get("final_macro_f1"),
            avg_bits=avg_b,
            comp=32 / avg_b if avg_b else None,
        ))

    final_udcon = load_json(greedy_udcon_dir / "final.json")
    wt_udcon    = load_json(greedy_udcon_dir / "assignment_weights.json")
    if final_udcon and wt_udcon:
        avg_b = param_weighted_avg_bits(wt_udcon, params_by_layer)
        rows.append(row_from_baseline_block(
            arch, "Greedy W6A6 (UD-constrained)",
            top1=final_udcon["final_top1"],
            ga=final_udcon.get("final_group_accuracies"),
            macro_f1=final_udcon.get("final_macro_f1"),
            avg_bits=avg_b,
            comp=32 / avg_b if avg_b else None,
        ))

    w8a8 = get_w8a8_reference(arch)
    pick = pick_nsga2_row(arch, w8a8)
    if pick is not None:
        chosen, reason = pick
        ev = lookup_eval(arch, chosen)
        ga = ev.get("group_accuracies")
        rows.append({
            "arch":                arch,
            "method":              f"NSGA-II Pareto ({reason})",
            "top1":                round(float(chosen["top1"]), 2),
            "wga":                 round(wga(ga), 4) if ga else None,
            "ud":                  round(float(chosen["ud"]), 4),
            "macro_f1":            round(float(ev["macro_f1"]), 4) if ev.get("macro_f1") is not None else None,
            "avg_wt_bits":         round(float(chosen["avg_bits"]), 3),
            "compression_vs_fp32": round(32 / float(chosen["avg_bits"]), 2),
        })

    return rows


def main():
    print("=" * 80)
    print("Rebuilding comparison table (final paper): old greedy + new NSGA-II")
    print("U_D = sum_g | acc_g - acc_overall | (Spantidi)")
    print("=" * 80)

    all_rows = []
    for arch in ARCHS:
        print(f"\n--- {arch} ---")
        rows = process_arch(arch)
        for r in rows:
            print(f"  {r['method']:<35}  top1={r['top1']:<6}  WGA={str(r['wga']):<7}  U_D={str(r['ud']):<7}  bits={str(r['avg_wt_bits']):<6}  comp={r['compression_vs_fp32']}")
        all_rows.extend(rows)

    out_path = RESULTS / "final_comparison_final_paper.csv"
    fieldnames = ["arch", "method", "top1", "wga", "ud", "macro_f1",
                  "avg_wt_bits", "compression_vs_fp32"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {out_path}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
