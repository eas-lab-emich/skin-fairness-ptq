"""
build_pareto_highlights.py
==========================
Curate 3 highlight Pareto solutions per architecture for paper discussion /
interpretability / reproducibility.

Per arch, pick 1 solution per objective:
  - best_top1     : highest top-1 accuracy
  - best_ud       : lowest unfairness disparity
  - best_avg_bits : lowest avg parameter-weighted bitwidth

Filter applied first: only consider solutions that DOMINATE W8A8 on all 3
objectives (≥ top1, ≤ ud, ≤ avg_bits, with at least one strict). If no such
solutions exist for an arch, fall back to the unfiltered Pareto (flagged in
output + console).

If the same solution wins multiple criteria (e.g. best_top1 is also best_ud),
it appears once with all criteria listed in `selected_for` (flagged in console).

Cross-references: each entry includes pareto_rank — the 1-based index in
pareto_front.json sorted by top1 desc, which matches the row in
energy_results.xlsx Pareto sheets.

W8A8 baselines pulled from final_comparison.csv.

Output: search/results/pareto_highlights.json
"""

import json
import os

RESULTS = '/home/mussa/skin-fairness-ptq/search/results'
SUFFIX  = '_final_paper'  # set to '' for legacy nsga2_<arch>_s42 dirs

PARETO_FILES = {
    arch: os.path.join(RESULTS, f'nsga2_{arch}_s42{SUFFIX}/pareto_front.json')
    for arch in ['vgg11', 'resnet18', 'resnet34', 'resnet50',
                 'mobilenet_v2', 'shufflenet_v2_x1_0']
}
OUT_PATH = os.path.join(RESULTS, f'pareto_highlights{SUFFIX}.json')

# W8A8 baselines from uniform_sweep_<arch>_s42/uniform_W8A8.json
# (top1 + Spantidi U_D = unfairness_score), avoids the mixed-units issue
# in the legacy final_comparison.csv.
W8A8 = {}
for arch in PARETO_FILES:
    with open(os.path.join(RESULTS, f'uniform_sweep_{arch}_s42/uniform_W8A8.json')) as f:
        u = json.load(f)
    W8A8[arch] = {'top1':     float(u['top1']),
                  'ud':       float(u['unfairness_score']),
                  'avg_bits': 8.0}


def with_rank(pareto):
    """Sort by top1 desc (matches Excel) and tag each entry with its rank."""
    s = sorted(pareto, key=lambda x: -x['top1'])
    for i, sol in enumerate(s):
        sol['pareto_rank'] = i + 1
    return s


def beats_w8a8(sol, w8):
    if not (sol['top1'] >= w8['top1'] and sol['ud'] <= w8['ud'] and sol['avg_bits'] <= w8['avg_bits']):
        return False
    return (sol['top1'] > w8['top1']) or (sol['ud'] < w8['ud']) or (sol['avg_bits'] < w8['avg_bits'])


def make_entry(sol, selected_for):
    return {
        'selected_for':    selected_for,
        'pareto_rank':     sol['pareto_rank'],
        'top1':            round(sol['top1'], 2),
        'ud':              round(sol['ud'], 4),
        'avg_bits':        round(sol['avg_bits'], 2),
        'assignment_wts':  sol['assignment_wts'],
        'assignment_acts': sol['assignment_acts'],
    }


fallback_archs = []
duplicate_notes = []
out = {}

for arch, pf_path in PARETO_FILES.items():
    with open(pf_path) as f:
        pareto = json.load(f)
    pareto = with_rank(pareto)
    w8 = W8A8[arch]

    dominators = [s for s in pareto if beats_w8a8(s, w8)]
    if dominators:
        pool = dominators
        filt = 'beats_w8a8'
    else:
        pool = pareto
        filt = 'unfiltered (no W8A8 dominators)'
        fallback_archs.append(arch)

    pick_top1 = max(pool, key=lambda s:  s['top1'])
    pick_ud   = min(pool, key=lambda s:  s['ud'])
    pick_bits = min(pool, key=lambda s:  s['avg_bits'])

    # Group duplicates by pareto_rank
    bucket = {}
    for label, sol in [('best_top1', pick_top1), ('best_ud', pick_ud), ('best_avg_bits', pick_bits)]:
        r = sol['pareto_rank']
        if r in bucket:
            bucket[r]['labels'].append(label)
        else:
            bucket[r] = {'sol': sol, 'labels': [label]}

    picks = [make_entry(b['sol'], b['labels']) for b in bucket.values()]
    picks.sort(key=lambda e: e['pareto_rank'])

    for entry in picks:
        if len(entry['selected_for']) > 1:
            duplicate_notes.append(
                f'{arch}: rank {entry["pareto_rank"]} won {entry["selected_for"]}'
            )

    out[arch] = {
        'filter':         filt,
        'w8a8_baseline':  {'top1': w8['top1'], 'ud': w8['ud'], 'avg_bits': 8.0},
        'n_dominators':   len(dominators),
        'picks':          picks,
    }

    print(f'{arch:20s}  pool={filt:35s}  '
          f'dominators={len(dominators):2d}  unique_picks={len(picks)}')


with open(OUT_PATH, 'w') as f:
    json.dump(out, f, indent=2)

print(f'\nWrote {OUT_PATH}  ({os.path.getsize(OUT_PATH):,} bytes)')

if fallback_archs:
    print(f'\n⚠ Fell back to unfiltered Pareto (no W8A8 dominators) for: {fallback_archs}')
else:
    print('\n✓ All archs had W8A8 dominators — no fallback needed.')

if duplicate_notes:
    print('\n⚠ Duplicate picks (same solution won multiple criteria):')
    for n in duplicate_notes:
        print(f'  {n}')
else:
    print('\n✓ All picks unique — no duplicates.')
