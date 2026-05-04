#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mussa/skin-fairness-ptq/backend"
PYTHON_BIN="/home/mussa/skin-fairness-ptq/.venv/bin/python"
SEARCH="$ROOT/search.py"
OUT_ROOT="$ROOT/results/smoke"

mkdir -p "$OUT_ROOT"

check_final() {
  local run_dir="$1"
  local final_json="$run_dir/final.json"
  if [[ ! -f "$final_json" ]]; then
    echo "Missing final.json: $final_json"
    exit 1
  fi
  "$PYTHON_BIN" - <<PY
import json
from pathlib import Path
f = Path("$final_json")
d = json.loads(f.read_text())
baseline = float(d["baseline_top1"])
final = float(d["final_top1"])
if not d.get("within_budget", False):
    raise SystemExit(f"within_budget=false in {f}")
if final < baseline - 10.0:
    raise SystemExit(f"catastrophic drop in {f}: baseline={baseline}, final={final}")
print(f"PASS {f}: baseline={baseline:.3f} final={final:.3f}")
PY
}

echo "[1/3] baseline+default PTQ sanity"
RUN1="$OUT_ROOT/run_sanity"
rm -rf "$RUN1"
"$PYTHON_BIN" "$SEARCH" \
  --search-mode weights \
  --eval-batches 1 \
  --refinement-passes 0 \
  --max-probe-layers 2 \
  --output-dir "$RUN1"
check_final "$RUN1"

echo "[2/3] weights-only search"
RUN2="$OUT_ROOT/run_weights_only"
rm -rf "$RUN2"
"$PYTHON_BIN" "$SEARCH" \
  --search-mode weights \
  --eval-batches 1 \
  --refinement-passes 0 \
  --max-probe-layers 4 \
  --output-dir "$RUN2"
check_final "$RUN2"

echo "[3/3] weights+activations search"
RUN3="$OUT_ROOT/run_weights_acts"
rm -rf "$RUN3"
"$PYTHON_BIN" "$SEARCH" \
  --search-mode weights_acts \
  --eval-batches 1 \
  --refinement-passes 0 \
  --max-probe-layers 4 \
  --bit-options-acts "8,7,6" \
  --output-dir "$RUN3"
check_final "$RUN3"

echo "Smoke test completed successfully"
