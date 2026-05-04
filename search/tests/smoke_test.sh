#!/usr/bin/env bash
# Smoke test for greedy_search.py
# Uses --eval-batches 1 and --budget-ratio 8.0 to keep runtime short (~3-5 min).
# Verifies the script runs end-to-end and produces all expected output files.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SKIN_CSV="/home/mussa/skin_fairness_project/data/fitzpatrick17k.csv"
SKIN_IMG="/home/mussa/fitzpatrick17k_data/data/finalfitz17k"
OUT_DIR="$SCRIPT_DIR/results/smoke/run_greedy_smoke"

echo "=== Greedy Fairness Smoke Test ==="
echo "Script: $SCRIPT_DIR/greedy_search.py"
echo "Output: $OUT_DIR"

python3 "$SCRIPT_DIR/greedy_search.py" \
    --skin-csv "$SKIN_CSV" \
    --skin-image-dir "$SKIN_IMG" \
    --eval-batches 1 \
    --budget-ratio 8.0 \
    --start-bits-wts 4 \
    --start-bits-acts 4 \
    --bit-options-wts "8,7,6,5,4" \
    --bit-options-acts "8,7,6,5,4" \
    --plateau-patience 0 \
    --seed 42 \
    --output-dir "$OUT_DIR"

echo ""
echo "=== Checking expected output files ==="
for f in baseline.json ptq_default.json ptq_start.json fairness_search_history.json \
          assignment_weights.json assignment_activations.json final.json \
          layer_inventory.json eval_cache.json run.log; do
    if [ -f "$OUT_DIR/$f" ]; then
        echo "  [OK] $f"
    else
        echo "  [MISSING] $f"
        exit 1
    fi
done

echo ""
echo "=== Checking fairness_search_history.json is valid JSON ==="
python3 -c "
import json, sys
with open('$OUT_DIR/fairness_search_history.json') as f:
    h = json.load(f)
print(f'  [OK] history has {len(h)} iteration(s)')
if h:
    required = ['iteration','layer','component','wga_before','wga_after','delta_wga',
                'ratio','compression_ratio_after','top1_after']
    for key in required:
        if key not in h[0]:
            print(f'  [MISSING key] {key} in first history entry')
            sys.exit(1)
        print(f'  [OK] key: {key}')
"

echo ""
echo "=== Checking final.json has key fields ==="
python3 -c "
import json, sys
with open('$OUT_DIR/final.json') as f:
    d = json.load(f)
required = ['baseline_top1','ptq_default_top1','ptq_start_top1','final_top1',
            'baseline_wga','ptq_start_wga','ptq_default_wga','final_wga',
            'compression_ratio_vs_fp32_weights','budget_ratio','stopping_reason',
            'total_iterations','search_algorithm']
for key in required:
    if key not in d:
        print(f'  [MISSING key] {key} in final.json')
        sys.exit(1)
    print(f'  [OK] {key} = {d[key]}')
"

echo ""
echo "=== Smoke test PASSED ==="
