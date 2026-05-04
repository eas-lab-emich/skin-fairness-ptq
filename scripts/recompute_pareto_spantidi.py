"""
recompute_pareto_spantidi.py
=============================
Recompute NSGA-II Pareto fronts using Spantidi U_D
  U_D = sum_g | acc_g - acc_overall |

The original NSGA-II runs minimised max-min UD ( max_g acc_g - min_g acc_g )
which is a different fairness metric. Every individual NSGA-II evaluated has
its full backend metrics (including the Spantidi U_D = unfairness_score and
the per-group accuracies) cached on disk, so we can re-extract a Pareto
front in (-top1, U_D_spantidi, avg_bits) space without re-running the search.

Caveat for the paper: the SEARCH was guided by max-min UD, so the candidate
pool is biased toward minimising that metric. The recomputed Pareto front is
the best Spantidi-U_D solutions found inside that biased pool. If the gap
between metrics is small enough, the front is still useful; if not, NSGA-II
needs a re-run.

Outputs per arch:
  results/nsga2_<arch>_s42/pareto_front_spantidi.json
"""

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


def assignment_key(wts, acts):
    return (json.dumps(wts, sort_keys=True), json.dumps(acts, sort_keys=True))


def load_layer_inventory(arch):
    """Layer inventory is written by greedy runs. NSGA-II uses the same model
    so params_by_layer is identical."""
    for prefix in ("greedy_w6a6", "greedy_ud_constrained"):
        path = RESULTS / f"{prefix}_{arch}_s42" / "layer_inventory.json"
        if path.exists():
            return json.loads(path.read_text())
    return None


def avg_weight_bits(assignment_wts, params_by_layer):
    """Parameter-weighted average over WEIGHT bits only.
    Mirrors run_nsga2.avg_bits which iterates the first L genes (= weights)."""
    total = sum(params_by_layer.values())
    if total == 0:
        return 0.0
    weighted = sum(int(assignment_wts[l]) * int(p) for l, p in params_by_layer.items())
    return weighted / total


def build_cache_lookup(cache_dict):
    """Map (frozen_wts, frozen_acts) -> cached metrics dict."""
    lookup = {}
    for entry in cache_dict.values():
        if "assignment_wts" not in entry:
            continue
        key = assignment_key(entry["assignment_wts"], entry["assignment_acts"])
        lookup[key] = entry
    return lookup


def non_dominated(items, key):
    """Naive O(N^2) 3D non-dominated set. Minimises every component of key(item)."""
    n = len(items)
    F = [tuple(key(it)) for it in items]
    keep = []
    for i in range(n):
        p = F[i]
        dominated = False
        for j in range(n):
            if i == j:
                continue
            q = F[j]
            if (q[0] <= p[0] and q[1] <= p[1] and q[2] <= p[2]
                    and (q[0] < p[0] or q[1] < p[1] or q[2] < p[2])):
                dominated = True
                break
        if not dominated:
            keep.append(items[i])
    return keep


def process_arch(arch):
    run_dir = RESULTS / f"nsga2_{arch}_s42"
    if not run_dir.exists():
        return {"arch": arch, "skipped": "no_run_dir"}

    all_path   = run_dir / "all_candidates.json"
    cache_path = run_dir / "eval_cache.json"
    if not (all_path.exists() and cache_path.exists()):
        return {"arch": arch, "skipped": "missing_inputs"}

    inv = load_layer_inventory(arch)
    if inv is None:
        return {"arch": arch, "skipped": "no_layer_inventory"}
    params_by_layer = {e["name"]: int(e["weight_params"]) for e in inv["layers"]}

    all_candidates = json.loads(all_path.read_text())
    cache          = json.loads(cache_path.read_text())
    lookup         = build_cache_lookup(cache)

    enriched = []
    seen_keys = set()
    misses = 0
    duplicates = 0
    for c in all_candidates:
        wts  = c["assignment_wts"]
        acts = c["assignment_acts"]
        key  = assignment_key(wts, acts)

        if key in seen_keys:
            duplicates += 1
            continue
        seen_keys.add(key)

        m = lookup.get(key)
        if m is None:
            misses += 1
            continue

        ga = m.get("group_accuracies", {})
        vals = [float(v) for v in ga.values()]
        if not vals:
            continue

        top1         = float(m["top1"])
        ud_spantidi  = float(m["unfairness_score"])
        ud_maxmin    = max(vals) - min(vals)
        avg_b        = avg_weight_bits(wts, params_by_layer)
        wga          = min(vals)
        macro_f1     = float(m.get("macro_f1", 0.0))

        enriched.append({
            "top1":            top1,
            "ud_spantidi":     ud_spantidi,
            "ud_maxmin":       ud_maxmin,
            "avg_bits":        avg_b,
            "wga":             wga,
            "macro_f1":        macro_f1,
            "assignment_wts":  wts,
            "assignment_acts": acts,
        })

    # Pareto front in Spantidi space: minimise (-top1, ud_spantidi, avg_bits)
    pareto = non_dominated(
        enriched,
        key=lambda r: (-r["top1"], r["ud_spantidi"], r["avg_bits"]),
    )
    pareto.sort(key=lambda r: -r["top1"])

    # Compare to original (max-min) Pareto
    orig_path = run_dir / "pareto_front.json"
    orig = json.loads(orig_path.read_text()) if orig_path.exists() else []
    orig_keys = {assignment_key(p["assignment_wts"], p["assignment_acts"]) for p in orig}
    new_keys  = {assignment_key(r["assignment_wts"], r["assignment_acts"]) for r in pareto}
    overlap = orig_keys & new_keys

    out = {
        "arch": arch,
        "metric": "U_D_spantidi = sum_g | acc_g - acc_overall |",
        "n_candidates_total": len(all_candidates),
        "n_unique_assignments": len(seen_keys),
        "n_enriched": len(enriched),
        "cache_misses": misses,
        "duplicates_skipped": duplicates,
        "pareto_size_spantidi": len(pareto),
        "pareto_size_maxmin_original": len(orig),
        "overlap_with_original_pareto": len(overlap),
        "pareto": pareto,
    }

    out_path = run_dir / "pareto_front_spantidi.json"
    out_path.write_text(json.dumps(out, indent=2))

    summary = {
        "arch": arch,
        "n_candidates": len(all_candidates),
        "n_unique": len(seen_keys),
        "pareto_spantidi": len(pareto),
        "pareto_original": len(orig),
        "overlap": len(overlap),
    }
    if pareto:
        best_top1 = max(pareto, key=lambda r: r["top1"])
        best_ud   = min(pareto, key=lambda r: r["ud_spantidi"])
        best_bits = min(pareto, key=lambda r: r["avg_bits"])
        summary["best_top1"] = {
            "top1":        round(best_top1["top1"], 2),
            "ud_spantidi": round(best_top1["ud_spantidi"], 4),
            "ud_maxmin":   round(best_top1["ud_maxmin"], 4),
            "avg_bits":    round(best_top1["avg_bits"], 2),
        }
        summary["best_ud"] = {
            "top1":        round(best_ud["top1"], 2),
            "ud_spantidi": round(best_ud["ud_spantidi"], 4),
            "ud_maxmin":   round(best_ud["ud_maxmin"], 4),
            "avg_bits":    round(best_ud["avg_bits"], 2),
        }
        summary["best_bits"] = {
            "top1":        round(best_bits["top1"], 2),
            "ud_spantidi": round(best_bits["ud_spantidi"], 4),
            "ud_maxmin":   round(best_bits["ud_maxmin"], 4),
            "avg_bits":    round(best_bits["avg_bits"], 2),
        }
    return summary


def main():
    print("=" * 78)
    print("Recomputing NSGA-II Pareto fronts in Spantidi U_D space")
    print("U_D = sum_g | acc_g - acc_overall |")
    print("=" * 78)
    summaries = []
    for arch in ARCHS:
        print(f"\n--- {arch} ---")
        s = process_arch(arch)
        summaries.append(s)
        if "skipped" in s:
            print(f"  SKIPPED: {s['skipped']}")
            continue
        print(f"  candidates={s['n_candidates']}  unique={s['n_unique']}")
        print(f"  Pareto (Spantidi)={s['pareto_spantidi']}  Pareto (max-min orig)={s['pareto_original']}  overlap={s['overlap']}")
        if "best_top1" in s:
            print(f"  best top1     : {s['best_top1']}")
            print(f"  best U_D      : {s['best_ud']}")
            print(f"  best avg_bits : {s['best_bits']}")

    out = RESULTS / "pareto_spantidi_summary.json"
    out.write_text(json.dumps(summaries, indent=2))
    print(f"\nSummary written to {out}")


if __name__ == "__main__":
    main()
