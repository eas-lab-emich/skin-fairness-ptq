#!/usr/bin/env bash
set -euo pipefail

"/home/mussa/skin-fairness-ptq/.venv/bin/python" \
  "/home/mussa/skin-fairness-ptq/backend/tools/update_excel.py" \
  --results-root "/home/mussa/skin-fairness-ptq/backend/results" \
  --out "/home/mussa/skin-fairness-ptq/distiller_mixed_precision_results.xlsx"
