#!/usr/bin/env bash
set -euo pipefail

# ---- Paths (edit only if your layout changes) ----
export PYTHONPATH=/home/mussa/skin-fairness-ptq/distiller
PY=/home/mussa/skin-fairness-ptq/.venv/bin/python
DATA=/home/mussa/skin-fairness-ptq/data
CKPT=/home/mussa/skin-fairness-ptq/distiller/examples/classifier_compression/r56_train_best.pth.tar
CC=/home/mussa/skin-fairness-ptq/distiller/examples/classifier_compression/compress_classifier.py
OUTDIR=/home/mussa/skin-fairness-ptq/results

# Curated run assignments
W_ONLY_DIR=/home/mussa/skin-fairness-ptq/backend/results/run_prof_weights_eval10
W_ACT_DIR=/home/mussa/skin-fairness-ptq/backend/results/run_prof_weights_acts_eval10

WMAP_W="${W_ONLY_DIR}/assignment_weights.json"
WMAP="${W_ACT_DIR}/assignment_weights.json"
AMAP="${W_ACT_DIR}/assignment_activations.json"

mkdir -p "${OUTDIR}"

echo "==> Validating FULL TEST (CIFAR-10: 10k images)."
echo "==> Checkpoint: ${CKPT}"
echo "==> Data dir:   ${DATA}"
echo "==> Logs ->     ${OUTDIR}"
echo

# 1) Baseline FP32 full test
echo "==> [1/4] Baseline FP32 full test"
${PY} ${CC} --arch resnet56_cifar ${DATA} \
  --resume ${CKPT} --evaluate --deterministic -j 1 -p 100 \
  --eval-batches 1000 --name r56_baseline_full \
  | tee "${OUTDIR}/r56_baseline_full.log"
echo

# 2) PTQ default (W8/A8) full test
echo "==> [2/4] PTQ default (W8/A8) full test"
${PY} ${CC} --arch resnet56_cifar ${DATA} \
  --resume ${CKPT} --evaluate --quantize --deterministic -j 1 -p 100 \
  --eval-batches 1000 --name r56_ptq_default_full \
  | tee "${OUTDIR}/r56_ptq_default_full.log"
echo

# 3) Curated weights-only assignment full test
echo "==> [3/4] Curated WEIGHTS-ONLY assignment full test"
if [[ ! -f "${WMAP_W}" ]]; then
  echo "ERROR: Missing weights-only map: ${WMAP_W}" >&2
  exit 1
fi
${PY} ${CC} --arch resnet56_cifar ${DATA} \
  --resume ${CKPT} --evaluate --quantize --deterministic -j 1 -p 100 \
  --q-bits-wts-map "${WMAP_W}" \
  --eval-batches 1000 --name r56_prof_weights_full \
  | tee "${OUTDIR}/r56_prof_weights_full.log"
echo

# 4) Curated weights+acts assignment full test
echo "==> [4/4] Curated WEIGHTS+ACTS assignment full test"
if [[ ! -f "${WMAP}" ]]; then
  echo "ERROR: Missing weights map: ${WMAP}" >&2
  exit 1
fi
if [[ ! -f "${AMAP}" ]]; then
  echo "ERROR: Missing activations map: ${AMAP}" >&2
  exit 1
fi
${PY} ${CC} --arch resnet56_cifar ${DATA} \
  --resume ${CKPT} --evaluate --quantize --deterministic -j 1 -p 100 \
  --q-bits-wts-map "${WMAP}" --q-bits-acts-map "${AMAP}" \
  --eval-batches 1000 --name r56_prof_weights_acts_full \
  | tee "${OUTDIR}/r56_prof_weights_acts_full.log"
echo

# ---- Print a compact summary by parsing the log files ----
echo "==> Summary (parsed from logs):"
${PY} - <<'PY'
import re, pathlib
paths = [
"/home/mussa/skin-fairness-ptq/results/r56_baseline_full.log",
"/home/mussa/skin-fairness-ptq/results/r56_ptq_default_full.log",
"/home/mussa/skin-fairness-ptq/results/r56_prof_weights_full.log",
"/home/mussa/skin-fairness-ptq/results/r56_prof_weights_acts_full.log",
]
pat = re.compile(r"Top1:\s*([0-9.]+)\s+Top5:\s*([0-9.]+)\s+Loss:\s*([0-9.]+)")
for p in paths:
    txt = pathlib.Path(p).read_text(errors="ignore")
    m=None
    for mm in pat.finditer(txt):
        m=mm
    name = pathlib.Path(p).name
    if m:
        print(f"{name:32s} Top1={m.group(1):>6s}  Top5={m.group(2):>6s}  Loss={m.group(3):>6s}")
    else:
        print(f"{name:32s} NO_MATCH")
PY
