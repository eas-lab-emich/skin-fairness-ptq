#!/usr/bin/env bash
# =============================================================================
# run_greedy_w6a6_all.sh
# =============================================================================
# Runs W6A6 greedy fairness search sequentially across all 6 architectures.
# Each arch runs to completion before the next one starts.
# On any failure the script stops and reports which arch failed.
#
# Usage:
#   bash scripts/run_greedy_w6a6_all.sh
#   bash scripts/run_greedy_w6a6_all.sh --start-from resnet34   # resume from arch
#   bash scripts/run_greedy_w6a6_all.sh --dry-run               # print commands only
#
# Logs:  logs/greedy_w6a6_<arch>.log
# Results: search/results/greedy_w6a6_<arch>_s42/
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
START_FROM=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-from) START_FROM="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=1;        shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GREEDY_SCRIPT="$ROOT/search/greedy_search.py"
RESULTS_ROOT="$ROOT/search/results"
LOGS_DIR="$ROOT/logs"
CKPT_BASE="$ROOT/backend/skin_checkpoints"
SKIN_CSV="/home/mussa/skin_fairness_project/data/fitzpatrick17k.csv"
SKIN_IMAGE_DIR="/home/mussa/fitzpatrick17k_data/data/finalfitz17k"
DISTILLER_DIR="$ROOT/distiller"

BUDGET_RATIO=3.0
BUDGET_MODE="combined_proxy"
PLATEAU_PATIENCE=5
SEED=42
LABEL_SPACE="nine"

# ── Arch order: smallest → largest (fastest results first) ────────────────────
ARCHS=(vgg11 resnet18 resnet34 resnet50 mobilenet_v2 shufflenet_v2_x1_0)

mkdir -p "$LOGS_DIR"

# ── Helper ────────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_greedy() {
    local arch="$1"
    local ckpt="$CKPT_BASE/fitz17k_${LABEL_SPACE}_${arch}_best.pth"
    local outdir="$RESULTS_ROOT/greedy_w6a6_${arch}_s${SEED}"
    local logfile="$LOGS_DIR/greedy_w6a6_${arch}.log"

    if [[ ! -f "$ckpt" ]]; then
        log "ERROR: checkpoint not found: $ckpt"
        return 1
    fi

    # Skip if already completed
    if [[ -f "$outdir/final.json" ]]; then
        log "SKIP $arch — final.json already exists at $outdir"
        return 0
    fi

    log "START $arch | budget=${BUDGET_RATIO}x ${BUDGET_MODE} | patience=${PLATEAU_PATIENCE} | log=${logfile}"

    local cmd=(
        python3 "$GREEDY_SCRIPT"
        --arch "$arch"
        --checkpoint "$ckpt"
        --skin-csv "$SKIN_CSV"
        --skin-image-dir "$SKIN_IMAGE_DIR"
        --skin-label-space "$LABEL_SPACE"
        --distiller-dir "$DISTILLER_DIR"
        --start-bits-wts 6
        --start-bits-acts 6
        --bit-options-wts "8,7,6"
        --bit-options-acts "8,7,6"
        --budget-ratio "$BUDGET_RATIO"
        --budget-mode "$BUDGET_MODE"
        --plateau-patience "$PLATEAU_PATIENCE"
        --seed "$SEED"
        --output-dir "$outdir"
    )

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY-RUN: ${cmd[*]} 2>&1 | tee $logfile"
        return 0
    fi

    "${cmd[@]}" 2>&1 | tee "$logfile"
    local exit_code="${PIPESTATUS[0]}"

    if [[ "$exit_code" -ne 0 ]]; then
        log "FAILED $arch (exit $exit_code) — see $logfile"
        return "$exit_code"
    fi

    log "DONE $arch"
}

# ── Main loop ─────────────────────────────────────────────────────────────────
log "============================================================"
log "  W6A6 Greedy Search — all architectures"
log "  budget_ratio=${BUDGET_RATIO}  mode=${BUDGET_MODE}  patience=${PLATEAU_PATIENCE}"
log "  results → $RESULTS_ROOT/greedy_w6a6_<arch>_s${SEED}/"
log "============================================================"

SKIP=1
if [[ -z "$START_FROM" ]]; then
    SKIP=0
fi

TOTAL=${#ARCHS[@]}
IDX=0
FAILED=()

for arch in "${ARCHS[@]}"; do
    IDX=$((IDX + 1))

    # --start-from support
    if [[ "$SKIP" -eq 1 ]]; then
        if [[ "$arch" == "$START_FROM" ]]; then
            SKIP=0
        else
            log "SKIP $arch (before --start-from=$START_FROM)"
            continue
        fi
    fi

    log "--- Arch $IDX/$TOTAL: $arch ---"
    if ! run_greedy "$arch"; then
        FAILED+=("$arch")
        log "Stopping chain due to failure on $arch."
        break
    fi
done

log "============================================================"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    log "All architectures completed successfully."
else
    log "FAILED: ${FAILED[*]}"
    exit 1
fi
log "============================================================"
