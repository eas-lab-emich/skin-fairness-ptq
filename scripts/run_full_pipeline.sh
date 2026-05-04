#!/usr/bin/env bash
# =============================================================================
# run_full_pipeline.sh
# =============================================================================
# Run all 3 phases of the skin fairness quantization pipeline for one model:
#   Phase 1 — Train on Fitzpatrick17k (FP32 baseline)
#   Phase 2 — Uniform quantization sweep  (W8A8, W7A7, W6A6, W5A5, W4A4)
#   Phase 3 — Greedy fairness search  (starting from W6A6, W5A5, W4A4)
#
# Usage:
#   bash run_full_pipeline.sh --arch mobilenet_v2
#   bash run_full_pipeline.sh --arch resnet32 --epochs 60 --skip-train
#
# Required:
#   --arch <name>    Architecture (resnet18|resnet32|resnet56|mobilenet_v2|shufflenet_v2_x1_0|vgg11)
#
# Optional:
#   --epochs <int>        Training epochs (default: 60)
#   --label-space <str>   binary|nine (default: nine)
#   --seed <int>          Random seed (default: 42)
#   --skip-train          Skip Phase 1 (use existing checkpoint)
#   --skip-sweep          Skip Phase 2
#   --skip-greedy         Skip Phase 3
#   --checkpoint <path>   Override checkpoint path (used if --skip-train)
#   --eval-batches <int>  Batches per eval, 0=full (default: 0)
#   --budget-ratio-w6 <float>   Budget ratio for W6A6 greedy (default: 3.0)
#   --budget-ratio-w5 <float>   Budget ratio for W5A5 greedy (default: 3.5)
#   --budget-ratio-w4 <float>   Budget ratio for W4A4 greedy (default: 4.0)
#   --plateau-patience <int>    Greedy plateau patience (default: 5)
#
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
ARCH=""
EPOCHS=60
LABEL_SPACE="nine"
SEED=42
SKIP_TRAIN=0
SKIP_SWEEP=0
SKIP_GREEDY=0
CHECKPOINT=""
EVAL_BATCHES=0
BUDGET_RATIO_W6=3.0
BUDGET_RATIO_W5=3.5
BUDGET_RATIO_W4=4.0
PLATEAU_PATIENCE=5

SKIN_CSV="/home/mussa/skin_fairness_project/data/fitzpatrick17k.csv"
SKIN_IMAGE_DIR="/home/mussa/fitzpatrick17k_data/data/finalfitz17k"
DISTILLER_DIR="/home/mussa/skin-fairness-ptq/distiller"
CKPT_BASE="/home/mussa/skin-fairness-ptq/backend/skin_checkpoints"
TRAIN_SCRIPT="/home/mussa/skin-fairness-ptq/backend/train_skin_fitz17k.py"
SWEEP_SCRIPT="/home/mussa/skin-fairness-ptq/search/run_uniform_sweep.py"
GREEDY_SCRIPT="/home/mussa/skin-fairness-ptq/search/greedy_search.py"
RESULTS_ROOT="/home/mussa/skin-fairness-ptq/search/results"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch)             ARCH="$2";               shift 2 ;;
        --epochs)           EPOCHS="$2";             shift 2 ;;
        --label-space)      LABEL_SPACE="$2";        shift 2 ;;
        --seed)             SEED="$2";               shift 2 ;;
        --skip-train)       SKIP_TRAIN=1;            shift ;;
        --skip-sweep)       SKIP_SWEEP=1;            shift ;;
        --skip-greedy)      SKIP_GREEDY=1;           shift ;;
        --checkpoint)       CHECKPOINT="$2";         shift 2 ;;
        --eval-batches)     EVAL_BATCHES="$2";       shift 2 ;;
        --budget-ratio-w6)  BUDGET_RATIO_W6="$2";   shift 2 ;;
        --budget-ratio-w5)  BUDGET_RATIO_W5="$2";   shift 2 ;;
        --budget-ratio-w4)  BUDGET_RATIO_W4="$2";   shift 2 ;;
        --plateau-patience) PLATEAU_PATIENCE="$2";   shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$ARCH" ]]; then
    echo "ERROR: --arch is required."
    echo "Choices: resnet18 resnet32 resnet56 mobilenet_v2 shufflenet_v2_x1_0 vgg11"
    exit 1
fi

# Default checkpoint path (where Phase 1 saves to / Phase 2-3 read from)
if [[ -z "$CHECKPOINT" ]]; then
    CHECKPOINT="${CKPT_BASE}/fitz17k_${LABEL_SPACE}_${ARCH}_best.pth"
fi

echo "========================================================================"
echo "  Skin Fairness Pipeline  |  arch=${ARCH}  |  label_space=${LABEL_SPACE}"
echo "  seed=${SEED}  epochs=${EPOCHS}  eval_batches=${EVAL_BATCHES}"
echo "  checkpoint=${CHECKPOINT}"
echo "========================================================================"

# ── Phase 1: Train ────────────────────────────────────────────────────────────
if [[ "$SKIP_TRAIN" -eq 0 ]]; then
    echo ""
    echo ">>> Phase 1: Training ${ARCH} on Fitzpatrick17k (${EPOCHS} epochs)..."
    python3 "$TRAIN_SCRIPT" \
        --arch "$ARCH" \
        --skin-csv "$SKIN_CSV" \
        --skin-image-dir "$SKIN_IMAGE_DIR" \
        --label-space "$LABEL_SPACE" \
        --output-dir "$CKPT_BASE" \
        --epochs "$EPOCHS" \
        --seed "$SEED"
    echo ">>> Phase 1 complete. Checkpoint: ${CHECKPOINT}"
else
    echo ">>> Phase 1 skipped (--skip-train)."
    if [[ ! -f "$CHECKPOINT" ]]; then
        echo "ERROR: Checkpoint not found at ${CHECKPOINT}"
        exit 1
    fi
fi

# ── Phase 2: Uniform Sweep ────────────────────────────────────────────────────
if [[ "$SKIP_SWEEP" -eq 0 ]]; then
    echo ""
    echo ">>> Phase 2: Uniform quantization sweep for ${ARCH}..."
    python3 "$SWEEP_SCRIPT" \
        --arch "$ARCH" \
        --checkpoint "$CHECKPOINT" \
        --skin-csv "$SKIN_CSV" \
        --skin-image-dir "$SKIN_IMAGE_DIR" \
        --skin-label-space "$LABEL_SPACE" \
        --eval-batches "$EVAL_BATCHES" \
        --seed "$SEED" \
        --distiller-dir "$DISTILLER_DIR" \
        --output-dir "${RESULTS_ROOT}/uniform_sweep_${ARCH}_s${SEED}"
    echo ">>> Phase 2 complete."
else
    echo ">>> Phase 2 skipped (--skip-sweep)."
fi

# ── Phase 3: Greedy Search ────────────────────────────────────────────────────
if [[ "$SKIP_GREEDY" -eq 0 ]]; then
    GREEDY_COMMON_ARGS=(
        --checkpoint "$CHECKPOINT"
        --arch "$ARCH"
        --skin-csv "$SKIN_CSV"
        --skin-image-dir "$SKIN_IMAGE_DIR"
        --skin-label-space "$LABEL_SPACE"
        --eval-batches "$EVAL_BATCHES"
        --seed "$SEED"
        --distiller-dir "$DISTILLER_DIR"
        --budget-mode combined_proxy
        --plateau-patience "$PLATEAU_PATIENCE"
    )

    echo ""
    echo ">>> Phase 3a: Greedy W6A6 for ${ARCH} (budget=${BUDGET_RATIO_W6})..."
    python3 "$GREEDY_SCRIPT" \
        "${GREEDY_COMMON_ARGS[@]}" \
        --start-bits-wts 6 --start-bits-acts 6 \
        --budget-ratio "$BUDGET_RATIO_W6" \
        --output-dir "${RESULTS_ROOT}/greedy_w6a6_${ARCH}_s${SEED}"

    echo ""
    echo ">>> Phase 3b: Greedy W5A5 for ${ARCH} (budget=${BUDGET_RATIO_W5})..."
    python3 "$GREEDY_SCRIPT" \
        "${GREEDY_COMMON_ARGS[@]}" \
        --start-bits-wts 5 --start-bits-acts 5 \
        --budget-ratio "$BUDGET_RATIO_W5" \
        --output-dir "${RESULTS_ROOT}/greedy_w5a5_${ARCH}_s${SEED}"

    echo ""
    echo ">>> Phase 3c: Greedy W4A4 for ${ARCH} (budget=${BUDGET_RATIO_W4})..."
    python3 "$GREEDY_SCRIPT" \
        "${GREEDY_COMMON_ARGS[@]}" \
        --start-bits-wts 4 --start-bits-acts 4 \
        --budget-ratio "$BUDGET_RATIO_W4" \
        --output-dir "${RESULTS_ROOT}/greedy_w4a4_${ARCH}_s${SEED}"

    echo ""
    echo ">>> Phase 3 complete."
else
    echo ">>> Phase 3 skipped (--skip-greedy)."
fi

echo ""
echo "========================================================================"
echo "  Pipeline DONE for arch=${ARCH}"
echo "  Results in: ${RESULTS_ROOT}/"
echo "========================================================================"
