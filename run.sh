#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
DISTILLER_DIR="$ROOT_DIR/distiller"
DATA_DIR="$ROOT_DIR/data"
RESULTS_DIR="$ROOT_DIR/results"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"
TRAIN_OWN="${TRAIN_OWN:-0}"

mkdir -p "$DATA_DIR" "$RESULTS_DIR"

echo "[1/8] Creating virtual environment"
if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv --system-site-packages "$VENV_DIR"
fi

echo "[2/8] Installing dependencies"
"$PIP_BIN" install --upgrade pip
"$PIP_BIN" install --no-build-isolation \
  pyyaml \
  torchnet==0.0.4 \
  visdom==0.2.4 \
  tabulate \
  pandas \
  gitpython \
  pydot \
  tensorflow-cpu==2.17.1

echo "[3/8] Cloning Distiller fork"
if [[ -d "$DISTILLER_DIR" ]]; then
  rm -rf "$DISTILLER_DIR"
fi
git clone https://github.com/KeyKy/distiller.git "$DISTILLER_DIR"

echo "[4/8] Applying minimal runtime compatibility fixes"
"$PYTHON_BIN" - <<'PY'
from pathlib import Path

files_and_replacements = {
    "examples/classifier_compression/compress_classifier.py": [
        ("target = target.cuda(async=True)", "target = target.cuda(non_blocking=True)"),
        ("losses['objective_loss'].add(loss.data[0])", "losses['objective_loss'].add(loss.item())"),
        ("losses['regularizer_loss'].add(regularizer_loss.data[0])", "losses['regularizer_loss'].add(regularizer_loss.item())"),
        ("return self.no_grad.__exit__(self, exc_type, exc_val, exc_tb)", "return self.no_grad.__exit__(exc_type, exc_val, exc_tb)"),
        (
            "parser.add_argument('--quantize', action='store_true',\n                    help='Apply 8-bit quantization to model before evaluation')",
            "parser.add_argument('--quantize', action='store_true',\n                    help='Apply 8-bit quantization to model before evaluation')\nparser.add_argument('--q-bits-wts', type=int, default=8,\n                    help='Override weight bitwidth for --quantize (default: 8)')\nparser.add_argument('--q-bits-acts', type=int, default=8,\n                    help='Override activation bitwidth for --quantize (default: 8)')"
        ),
        (
            "quantizer = quantization.SymmetricLinearQuantizer(model, 8, 8)",
            "quantizer = quantization.SymmetricLinearQuantizer(model, args.q_bits_acts, args.q_bits_wts)"
        ),
    ],
    "distiller/data_loggers/tbbackend.py": [
        ("import tensorflow as tf", "import tensorflow.compat.v1 as tf"),
        ("import numpy as np", "import numpy as np\n\ntf.disable_v2_behavior()"),
    ],
    "distiller/quantization/q_utils.py": [
        ("return max(abs(tensor.max().data[0]), abs(tensor.min().data[0]))", "return max(abs(tensor.max().item()), abs(tensor.min().item()))"),
    ],
    "distiller/pruning/sensitivity_pruner.py": [
        ("param.stddev = torch.std(param).data[0]", "param.stddev = torch.std(param).item()"),
    ],
    "distiller/model_summaries.py": [
        ("pd.set_option('precision', 2)", "pd.set_option('display.precision', 2)"),
    ],
}

repo = Path("/home/mussa/skin-fairness-ptq/distiller")
for rel_path, replacements in files_and_replacements.items():
    path = repo / rel_path
    text = path.read_text()
    for old, new in replacements:
        text = text.replace(old, new)
    path.write_text(text)
PY

echo "[5/8] Verifying distiller import"
PYTHONPATH="$DISTILLER_DIR" "$PYTHON_BIN" -c "import distiller; print('distiller import: OK')"

HELP_LOG="$RESULTS_DIR/help.log"
echo "[6/8] Detecting quantization flag from --help"
HELP_OUTPUT="$(PYTHONPATH="$DISTILLER_DIR" "$PYTHON_BIN" "$DISTILLER_DIR/examples/classifier_compression/compress_classifier.py" --help 2>&1 | tee "$HELP_LOG")"

QUANT_FLAG=""
if grep -q -- "--quantize-eval" <<< "$HELP_OUTPUT"; then
  QUANT_FLAG="--quantize-eval"
elif grep -q -- "--quantize" <<< "$HELP_OUTPUT"; then
  QUANT_FLAG="--quantize"
else
  echo "ERROR: Could not detect quantization flag from --help output."
  exit 1
fi

EXAMPLE_DIR="$DISTILLER_DIR/examples/classifier_compression"
CHECKPOINT_PATH=""
BASELINE_LOG=""
PTQ_LOG=""
PTQ_REPEAT_LOG=""
PTQ_W4A4_LOG=""
PTQ_W4A8_LOG=""
PTQ_W8A4_LOG=""
TRAIN_LOG=""
PTQ_W4A4_CMD=""
PTQ_W4A8_CMD=""
PTQ_W8A4_CMD=""

if [[ "$TRAIN_OWN" == "1" ]]; then
  echo "[7/8] TRAIN_OWN=1: Training resnet56_cifar for 200 epochs"
  ARCH="resnet56_cifar"
  TRAIN_LOG="$RESULTS_DIR/r56_train.log"
  BASELINE_LOG="$RESULTS_DIR/r56_baseline_eval.log"
  PTQ_LOG="$RESULTS_DIR/r56_ptq_eval.log"
  PTQ_REPEAT_LOG="$RESULTS_DIR/r56_ptq_eval_repeat.log"
  PTQ_W4A4_LOG="$RESULTS_DIR/r56_ptq_eval_w4a4.log"
  PTQ_W4A8_LOG="$RESULTS_DIR/r56_ptq_eval_w4a8.log"
  PTQ_W8A4_LOG="$RESULTS_DIR/r56_ptq_eval_w8a4.log"
  rm -f "$TRAIN_LOG" "$BASELINE_LOG" "$PTQ_LOG" "$PTQ_REPEAT_LOG" "$PTQ_W4A4_LOG" "$PTQ_W4A8_LOG" "$PTQ_W8A4_LOG"

  TRAIN_LR_ARGS=()
  if grep -Eq -- '--lr-decay-epochs( |$|=)' <<< "$HELP_OUTPUT"; then
    TRAIN_LR_ARGS+=(--lr-decay-epochs 100,150)
    if grep -Eq -- '--lr-decay( |$|=)' <<< "$HELP_OUTPUT"; then
      TRAIN_LR_ARGS+=(--lr-decay 0.1)
    fi
    echo "Detected milestone LR flags: ${TRAIN_LR_ARGS[*]}"
  elif grep -Eq -- '--lr-schedule( |$|=)' <<< "$HELP_OUTPUT"; then
    TRAIN_LR_ARGS+=(--lr-schedule 100,150)
    echo "Detected lr schedule flag: ${TRAIN_LR_ARGS[*]}"
  elif grep -Eq -- '--step( |$|=)' <<< "$HELP_OUTPUT"; then
    TRAIN_LR_ARGS+=(--step 100)
    if grep -Eq -- '--gamma( |$|=)' <<< "$HELP_OUTPUT"; then
      TRAIN_LR_ARGS+=(--gamma 0.1)
    fi
    echo "Detected step LR flags: ${TRAIN_LR_ARGS[*]}"
  else
    echo "No explicit LR milestone flag detected; using supported default scheduler behavior for 200 epochs."
  fi

  (
    cd "$EXAMPLE_DIR"
    PYTHONPATH="$DISTILLER_DIR" "$PYTHON_BIN" compress_classifier.py \
      --arch resnet56_cifar "$DATA_DIR" --epochs 200 --batch-size 128 --lr 0.1 --momentum 0.9 --weight-decay 1e-4 \
      --deterministic -j 1 -p 100 --name r56_train "${TRAIN_LR_ARGS[@]}"
  ) 2>&1 | tee "$TRAIN_LOG"

  TRAIN_FINAL_TOP1="$(TRAIN_LOG="$TRAIN_LOG" "$PYTHON_BIN" - <<'PY'
import os
import re
from pathlib import Path

text = Path(os.environ["TRAIN_LOG"]).read_text()
matches = re.findall(r"==> Top1:\s*([0-9.]+)\s+Top5:\s*([0-9.]+)\s+Loss:\s*([0-9.]+)", text)
if not matches:
    raise RuntimeError("Could not find final Top1 in training log")
top1, top5, loss = matches[-1]
print(f"TRAIN_FINAL_TOP1={top1} TRAIN_FINAL_TOP5={top5} TRAIN_FINAL_LOSS={loss}")
PY
)"
  echo "$TRAIN_FINAL_TOP1" | tee -a "$TRAIN_LOG"

  CHECKPOINT_PATH="$(TRAIN_LOG="$TRAIN_LOG" EXAMPLE_DIR="$EXAMPLE_DIR" "$PYTHON_BIN" - <<'PY'
import os
import re
from pathlib import Path

train_log = Path(os.environ["TRAIN_LOG"])
example_dir = Path(os.environ["EXAMPLE_DIR"])
text = train_log.read_text()

matches = re.findall(r"Log file for this run:\s*(.+)", text)
run_dir = None
if matches:
    run_dir = Path(matches[-1].strip()).parent

def scored_candidates(paths):
    scored = []
    for p in paths:
        name = p.name.lower()
        if not p.is_file():
            continue
        score = 0
        if "best" in name or name == "model_best.pth.tar":
            score += 100
        elif "checkpoint" in name:
            score += 50
        if "r56_train" in name:
            score += 10
        scored.append((score, p.stat().st_mtime, p))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [x[2] for x in scored]

candidates = []
if run_dir and run_dir.exists():
    candidates.extend(run_dir.rglob("*.pth.tar"))
candidates.extend(example_dir.glob("*.pth.tar"))

ordered = scored_candidates(candidates)
if not ordered:
    raise RuntimeError("No checkpoint files found after r56_train run")

print(str(ordered[0]))
PY
)"

  if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "ERROR: Could not find trained checkpoint path: $CHECKPOINT_PATH"
    exit 1
  fi

  BASELINE_CMD="PYTHONPATH=$DISTILLER_DIR $PYTHON_BIN compress_classifier.py --arch $ARCH $DATA_DIR --resume $CHECKPOINT_PATH --evaluate --deterministic -j 1 -p 100"
  PTQ_CMD="PYTHONPATH=$DISTILLER_DIR $PYTHON_BIN compress_classifier.py --arch $ARCH $DATA_DIR --resume $CHECKPOINT_PATH --evaluate $QUANT_FLAG --deterministic -j 1 -p 100"
  PTQ_W4A4_CMD="PYTHONPATH=$DISTILLER_DIR $PYTHON_BIN compress_classifier.py --arch $ARCH $DATA_DIR --resume $CHECKPOINT_PATH --evaluate $QUANT_FLAG --q-bits-wts 4 --q-bits-acts 4 --deterministic -j 1 -p 100"
  PTQ_W4A8_CMD="PYTHONPATH=$DISTILLER_DIR $PYTHON_BIN compress_classifier.py --arch $ARCH $DATA_DIR --resume $CHECKPOINT_PATH --evaluate $QUANT_FLAG --q-bits-wts 4 --q-bits-acts 8 --deterministic -j 1 -p 100"
  PTQ_W8A4_CMD="PYTHONPATH=$DISTILLER_DIR $PYTHON_BIN compress_classifier.py --arch $ARCH $DATA_DIR --resume $CHECKPOINT_PATH --evaluate $QUANT_FLAG --q-bits-wts 8 --q-bits-acts 4 --deterministic -j 1 -p 100"

  echo "[8/8] Running r56 baseline/PTQ/PTQ-repeat evals"
  (
    cd "$EXAMPLE_DIR"
    PYTHONPATH="$DISTILLER_DIR" "$PYTHON_BIN" compress_classifier.py \
      --arch "$ARCH" "$DATA_DIR" --resume "$CHECKPOINT_PATH" --evaluate --deterministic -j 1 -p 100
  ) 2>&1 | tee "$BASELINE_LOG"

  if ! grep -q "==> Top1:" "$BASELINE_LOG"; then
    echo "ERROR: baseline eval log does not contain '==> Top1:'"
    exit 1
  fi

  (
    cd "$EXAMPLE_DIR"
    PYTHONPATH="$DISTILLER_DIR" "$PYTHON_BIN" compress_classifier.py \
      --arch "$ARCH" "$DATA_DIR" --resume "$CHECKPOINT_PATH" --evaluate "$QUANT_FLAG" --deterministic -j 1 -p 100
  ) 2>&1 | tee "$PTQ_LOG"

  (
    cd "$EXAMPLE_DIR"
    PYTHONPATH="$DISTILLER_DIR" "$PYTHON_BIN" compress_classifier.py \
      --arch "$ARCH" "$DATA_DIR" --resume "$CHECKPOINT_PATH" --evaluate "$QUANT_FLAG" --deterministic -j 1 -p 100
  ) 2>&1 | tee "$PTQ_REPEAT_LOG"

  (
    cd "$EXAMPLE_DIR"
    PYTHONPATH="$DISTILLER_DIR" "$PYTHON_BIN" compress_classifier.py \
      --arch "$ARCH" "$DATA_DIR" --resume "$CHECKPOINT_PATH" --evaluate "$QUANT_FLAG" --q-bits-wts 4 --q-bits-acts 4 --deterministic -j 1 -p 100
  ) 2>&1 | tee "$PTQ_W4A4_LOG"

  (
    cd "$EXAMPLE_DIR"
    PYTHONPATH="$DISTILLER_DIR" "$PYTHON_BIN" compress_classifier.py \
      --arch "$ARCH" "$DATA_DIR" --resume "$CHECKPOINT_PATH" --evaluate "$QUANT_FLAG" --q-bits-wts 4 --q-bits-acts 8 --deterministic -j 1 -p 100
  ) 2>&1 | tee "$PTQ_W4A8_LOG"

  (
    cd "$EXAMPLE_DIR"
    PYTHONPATH="$DISTILLER_DIR" "$PYTHON_BIN" compress_classifier.py \
      --arch "$ARCH" "$DATA_DIR" --resume "$CHECKPOINT_PATH" --evaluate "$QUANT_FLAG" --q-bits-wts 8 --q-bits-acts 4 --deterministic -j 1 -p 100
  ) 2>&1 | tee "$PTQ_W8A4_LOG"
else
  echo "[7/8] TRAIN_OWN=0: Using shipped checkpoint mode"
  ARCH=""
  if grep -q "resnet50_cifar" <<< "$HELP_OUTPUT"; then
    ARCH="resnet50_cifar"
  elif grep -q "resnet20_cifar" <<< "$HELP_OUTPUT"; then
    ARCH="resnet20_cifar"
  else
    echo "ERROR: Could not find a CIFAR ResNet architecture in --help output."
    exit 1
  fi

  if [[ "$ARCH" == "resnet20_cifar" && -f "$DISTILLER_DIR/examples/ssl/checkpoints/checkpoint_trained_dense.pth.tar" ]]; then
    CHECKPOINT_PATH="$DISTILLER_DIR/examples/ssl/checkpoints/checkpoint_trained_dense.pth.tar"
  else
    echo "No shipped checkpoint for $ARCH found. Training a short baseline (3 epochs)."
    (
      cd "$EXAMPLE_DIR"
      PYTHONPATH="$DISTILLER_DIR" "$PYTHON_BIN" compress_classifier.py \
        --arch "$ARCH" "$DATA_DIR" --epochs 3 --deterministic -j 1 -p 100 --name short_baseline
    ) 2>&1 | tee "$RESULTS_DIR/baseline_train.log"
    CHECKPOINT_PATH="$EXAMPLE_DIR/short_baseline_checkpoint.pth.tar"
    if [[ ! -f "$CHECKPOINT_PATH" ]]; then
      echo "ERROR: Expected checkpoint not found at $CHECKPOINT_PATH"
      exit 1
    fi
  fi

  BASELINE_LOG="$RESULTS_DIR/baseline_eval.log"
  PTQ_LOG="$RESULTS_DIR/ptq_eval.log"
  PTQ_REPEAT_LOG="$RESULTS_DIR/ptq_eval.log"
  rm -f "$BASELINE_LOG" "$PTQ_LOG"

  BASELINE_CMD="PYTHONPATH=$DISTILLER_DIR $PYTHON_BIN compress_classifier.py --arch $ARCH $DATA_DIR --resume $CHECKPOINT_PATH --evaluate --deterministic -j 1 -p 100"
  PTQ_CMD="PYTHONPATH=$DISTILLER_DIR $PYTHON_BIN compress_classifier.py --arch $ARCH $DATA_DIR --resume $CHECKPOINT_PATH --evaluate $QUANT_FLAG --deterministic -j 1 -p 100"

  echo "[8/8] Running baseline eval"
  (
    cd "$EXAMPLE_DIR"
    PYTHONPATH="$DISTILLER_DIR" "$PYTHON_BIN" compress_classifier.py \
      --arch "$ARCH" "$DATA_DIR" --resume "$CHECKPOINT_PATH" --evaluate --deterministic -j 1 -p 100
  ) 2>&1 | tee "$BASELINE_LOG"

  if ! grep -q "==> Top1:" "$BASELINE_LOG"; then
    echo "ERROR: baseline eval log does not contain '==> Top1:'"
    exit 1
  fi

  echo "Running PTQ eval twice"
  echo "### PTQ RUN 1 ###" | tee -a "$PTQ_LOG"
  (
    cd "$EXAMPLE_DIR"
    PYTHONPATH="$DISTILLER_DIR" "$PYTHON_BIN" compress_classifier.py \
      --arch "$ARCH" "$DATA_DIR" --resume "$CHECKPOINT_PATH" --evaluate "$QUANT_FLAG" --deterministic -j 1 -p 100
  ) 2>&1 | tee -a "$PTQ_LOG"

  echo "### PTQ RUN 2 ###" | tee -a "$PTQ_LOG"
  (
    cd "$EXAMPLE_DIR"
    PYTHONPATH="$DISTILLER_DIR" "$PYTHON_BIN" compress_classifier.py \
      --arch "$ARCH" "$DATA_DIR" --resume "$CHECKPOINT_PATH" --evaluate "$QUANT_FLAG" --deterministic -j 1 -p 100
  ) 2>&1 | tee -a "$PTQ_LOG"
fi

BASELINE_JSON="$RESULTS_DIR/baseline_metrics.json"
PTQ_JSON="$RESULTS_DIR/ptq_metrics.json"

ARCH="$ARCH" QUANT_FLAG="$QUANT_FLAG" CHECKPOINT_PATH="$CHECKPOINT_PATH" BASELINE_CMD="$BASELINE_CMD" PTQ_CMD="$PTQ_CMD" PTQ_W4A4_CMD="$PTQ_W4A4_CMD" PTQ_W4A8_CMD="$PTQ_W4A8_CMD" PTQ_W8A4_CMD="$PTQ_W8A4_CMD" BASELINE_LOG="$BASELINE_LOG" PTQ_LOG="$PTQ_LOG" PTQ_REPEAT_LOG="$PTQ_REPEAT_LOG" PTQ_W4A4_LOG="$PTQ_W4A4_LOG" PTQ_W4A8_LOG="$PTQ_W4A8_LOG" PTQ_W8A4_LOG="$PTQ_W8A4_LOG" TRAIN_OWN="$TRAIN_OWN" "$PYTHON_BIN" - <<'PY'
import json
import os
import re
from pathlib import Path

root = Path("/home/mussa/skin-fairness-ptq")
results = root / "results"

def parse_metrics(path: Path):
    text = path.read_text()
    matches = re.findall(r"==> Top1:\s*([0-9.]+)\s+Top5:\s*([0-9.]+)\s+Loss:\s*([0-9.]+)", text)
    if not matches:
        raise RuntimeError(f"Could not parse metrics from {path}")
    return [tuple(map(float, m)) for m in matches]

baseline_log = Path(os.environ["BASELINE_LOG"])
ptq_log = Path(os.environ["PTQ_LOG"])
ptq_repeat_log = Path(os.environ["PTQ_REPEAT_LOG"])
ptq_w4a4_log = Path(os.environ.get("PTQ_W4A4_LOG", "")) if os.environ.get("PTQ_W4A4_LOG") else None
ptq_w4a8_log = Path(os.environ.get("PTQ_W4A8_LOG", "")) if os.environ.get("PTQ_W4A8_LOG") else None
ptq_w8a4_log = Path(os.environ.get("PTQ_W8A4_LOG", "")) if os.environ.get("PTQ_W8A4_LOG") else None
train_own = os.environ["TRAIN_OWN"] == "1"

baseline_all = parse_metrics(baseline_log)
baseline_top1, baseline_top5, baseline_loss = baseline_all[-1]

if train_own:
    ptq_all = parse_metrics(ptq_log)
    ptq_repeat_all = parse_metrics(ptq_repeat_log)
    ptq1_top1, ptq1_top5, ptq1_loss = ptq_all[-1]
    ptq2_top1, ptq2_top5, ptq2_loss = ptq_repeat_all[-1]
    if ptq_w4a4_log is None:
        raise RuntimeError("Expected PTQ_W4A4_LOG in TRAIN_OWN mode")
    if ptq_w4a8_log is None:
        raise RuntimeError("Expected PTQ_W4A8_LOG in TRAIN_OWN mode")
    if ptq_w8a4_log is None:
        raise RuntimeError("Expected PTQ_W8A4_LOG in TRAIN_OWN mode")
    ptq_w4a4_all = parse_metrics(ptq_w4a4_log)
    ptq_w4a4_top1, ptq_w4a4_top5, ptq_w4a4_loss = ptq_w4a4_all[-1]
    ptq_w4a8_all = parse_metrics(ptq_w4a8_log)
    ptq_w4a8_top1, ptq_w4a8_top5, ptq_w4a8_loss = ptq_w4a8_all[-1]
    ptq_w8a4_all = parse_metrics(ptq_w8a4_log)
    ptq_w8a4_top1, ptq_w8a4_top5, ptq_w8a4_loss = ptq_w8a4_all[-1]
else:
    ptq_all = parse_metrics(ptq_log)
    if len(ptq_all) < 2:
        raise RuntimeError("Expected two PTQ metric entries in ptq_eval.log")
    ptq1_top1, ptq1_top5, ptq1_loss = ptq_all[0]
    ptq2_top1, ptq2_top5, ptq2_loss = ptq_all[1]

checkpoint = os.environ["CHECKPOINT_PATH"]
arch = os.environ["ARCH"]
quant_flag = os.environ["QUANT_FLAG"]
baseline_cmd = os.environ["BASELINE_CMD"]
ptq_cmd = os.environ["PTQ_CMD"]
ptq_w4a4_cmd = os.environ.get("PTQ_W4A4_CMD", "")
ptq_w4a8_cmd = os.environ.get("PTQ_W4A8_CMD", "")
ptq_w8a4_cmd = os.environ.get("PTQ_W8A4_CMD", "")

baseline_payload = {
    "top1": baseline_top1,
    "top5": baseline_top5,
    "loss": baseline_loss,
    "checkpoint_path": checkpoint,
    "command": baseline_cmd,
}

ptq_payload = {
    "top1": ptq1_top1,
    "top5": ptq1_top5,
    "loss": ptq1_loss,
    "checkpoint_path_used": checkpoint,
    "command": ptq_cmd,
    "repeat_run": {
      "top1": ptq2_top1,
      "top5": ptq2_top5,
      "loss": ptq2_loss
    }
}

(results / "baseline_metrics.json").write_text(json.dumps(baseline_payload, indent=2) + "\n")
(results / "ptq_metrics.json").write_text(json.dumps(ptq_payload, indent=2) + "\n")

if train_own:
    ptq_w4a4_payload = {
        "top1": ptq_w4a4_top1,
        "top5": ptq_w4a4_top5,
        "loss": ptq_w4a4_loss,
        "checkpoint_path_used": checkpoint,
        "command": ptq_w4a4_cmd,
    }
    (results / "ptq_metrics_w4a4.json").write_text(json.dumps(ptq_w4a4_payload, indent=2) + "\n")

    ptq_w4a8_payload = {
        "top1": ptq_w4a8_top1,
        "top5": ptq_w4a8_top5,
        "loss": ptq_w4a8_loss,
        "checkpoint_path_used": checkpoint,
        "command": ptq_w4a8_cmd,
    }
    (results / "ptq_metrics_w4a8.json").write_text(json.dumps(ptq_w4a8_payload, indent=2) + "\n")

    ptq_w8a4_payload = {
        "top1": ptq_w8a4_top1,
        "top5": ptq_w8a4_top5,
        "loss": ptq_w8a4_loss,
        "checkpoint_path_used": checkpoint,
        "command": ptq_w8a4_cmd,
    }
    (results / "ptq_metrics_w8a4.json").write_text(json.dumps(ptq_w8a4_payload, indent=2) + "\n")

drop = baseline_top1 - ptq1_top1
print(f"FINAL SUMMARY: baseline Top1={baseline_top1:.3f}, PTQ Top1={ptq1_top1:.3f}, drop={drop:.3f}")
print(f"PTQ repeat check: run1 Top1={ptq1_top1:.3f}, run2 Top1={ptq2_top1:.3f}, delta={abs(ptq1_top1-ptq2_top1):.6f}")

rows = [("baseline", baseline_top1, 0.0), ("ptq_default", ptq1_top1, baseline_top1 - ptq1_top1)]
if train_own:
    rows.extend([
        ("ptq_w4a4", ptq_w4a4_top1, baseline_top1 - ptq_w4a4_top1),
        ("ptq_w4a8", ptq_w4a8_top1, baseline_top1 - ptq_w4a8_top1),
        ("ptq_w8a4", ptq_w8a4_top1, baseline_top1 - ptq_w8a4_top1),
    ])

print("config, top1, drop_vs_baseline")
for cfg, top1, drop_val in rows:
    print(f"{cfg}, {top1:.3f}, {drop_val:.3f}")
PY
