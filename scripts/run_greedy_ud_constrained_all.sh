#!/usr/bin/env bash
# =============================================================================
# run_greedy_ud_constrained_all.sh
# =============================================================================
# Runs UD-constrained W6A6 greedy fairness search across all 6 architectures.
# Each arch has a hard UD ceiling = W8A8_UD + 0.02 (candidates exceeding this
# are rejected). UD here is Spantidi U_D = sum_g |acc_g - acc_overall|, which
# matches the backend's `unfairness_score` field. (Earlier ceilings were
# derived from max-min UD and applied against Spantidi U_D, which silently
# made every probe ineligible — fixed below.) All evaluated candidates are
# logged to all_candidates_log.json for Pareto front analysis.
#
# UD ceilings used (Spantidi W8A8 UD + 0.02):
#   vgg11              0.229 + 0.02 = 0.249
#   resnet18           0.229 + 0.02 = 0.249
#   resnet34           0.142 + 0.02 = 0.162
#   resnet50           0.172 + 0.02 = 0.192
#   mobilenet_v2       0.181 + 0.02 = 0.201
#   shufflenet_v2_x1_0 0.195 + 0.02 = 0.215
#
# Usage:
#   bash scripts/run_greedy_ud_constrained_all.sh
#   bash scripts/run_greedy_ud_constrained_all.sh --start-from resnet50
#   bash scripts/run_greedy_ud_constrained_all.sh --dry-run
#
# Logs:   logs/greedy_ud_constrained_<arch>.log
# Results: search/results/greedy_ud_constrained_<arch>_s42/
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

# ── Per-arch UD ceilings (Spantidi W8A8 U_D + 0.02) ──────────────────────────
declare -A MAX_UD
MAX_UD[vgg11]=0.249
MAX_UD[resnet18]=0.249
MAX_UD[resnet34]=0.162
MAX_UD[resnet50]=0.192
MAX_UD[mobilenet_v2]=0.201
MAX_UD[shufflenet_v2_x1_0]=0.215

# ── Arch order: smallest → largest (fastest results first) ────────────────────
ARCHS=(vgg11 mobilenet_v2 shufflenet_v2_x1_0 resnet18 resnet34 resnet50)

mkdir -p "$LOGS_DIR"

# ── Helper ────────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_greedy() {
    local arch="$1"
    local max_ud="${MAX_UD[$arch]}"
    local ckpt="$CKPT_BASE/fitz17k_${LABEL_SPACE}_${arch}_best.pth"
    local outdir="$RESULTS_ROOT/greedy_ud_constrained_${arch}_s${SEED}_final_paper"
    local logfile="$LOGS_DIR/greedy_ud_constrained_${arch}_final_paper.log"

    if [[ ! -f "$ckpt" ]]; then
        log "ERROR: checkpoint not found: $ckpt"
        return 1
    fi

    # Skip if already completed
    if [[ -f "$outdir/final.json" ]]; then
        log "SKIP $arch — final.json already exists at $outdir"
        return 0
    fi

    log "START $arch | max_ud=${max_ud} | budget=${BUDGET_RATIO}x ${BUDGET_MODE} | patience=${PLATEAU_PATIENCE} | log=${logfile}"

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
        --max-ud "$max_ud"
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
log "  UD-Constrained W6A6 Greedy Search — all architectures"
log "  budget_ratio=${BUDGET_RATIO}  mode=${BUDGET_MODE}  patience=${PLATEAU_PATIENCE}"
log "  Spantidi U_D ceilings: vgg11=0.249 resnet18=0.249 resnet34=0.162"
log "                         resnet50=0.192 mobilenet_v2=0.201 shufflenet=0.215"
log "  results → $RESULTS_ROOT/greedy_ud_constrained_<arch>_s${SEED}_final_paper/"
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
