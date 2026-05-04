#!/usr/bin/env bash
# =============================================================================
# run_nsga2_queue_final_paper.sh
# =============================================================================
# Sequential NSGA-II rerun across all 6 architectures, with the corrected
# Spantidi U_D objective (run_nsga2.py:190 patch). Outputs land in
# nsga2_<arch>_s42_final_paper/ and logs in nsga2_<arch>_final_paper.log so
# the original max-min runs in nsga2_<arch>_s42/ remain as audit trail.
#
# Usage:
#   bash scripts/run_nsga2_queue_final_paper.sh
#   bash scripts/run_nsga2_queue_final_paper.sh --start-from resnet18
#   bash scripts/run_nsga2_queue_final_paper.sh --dry-run
#
# Recommended invocation (detached tmux):
#   tmux new-session -d -s nsga2_rerun \
#     "bash scripts/run_nsga2_queue_final_paper.sh"
#   tmux attach -t nsga2_rerun     # to watch
# =============================================================================

set -euo pipefail

START_FROM=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-from) START_FROM="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=1;        shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

ROOT=/home/mussa/skin-fairness-ptq
RESULTS=$ROOT/search/results
LOGS=$ROOT/logs
SUFFIX=_final_paper

mkdir -p "$LOGS"

# Smallest → largest for fastest feedback on the early archs.
ARCHS=(vgg11 mobilenet_v2 shufflenet_v2_x1_0 resnet18 resnet34 resnet50)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "============================================================"
log "  NSGA-II rerun (Spantidi U_D objective, --out-suffix=$SUFFIX)"
log "  pop=50  gens=30  seed=42"
log "  results → $RESULTS/nsga2_<arch>_s42${SUFFIX}/"
log "  logs    → $LOGS/nsga2_<arch>${SUFFIX}.log"
log "============================================================"

SKIP=1
[[ -z "$START_FROM" ]] && SKIP=0

FAILED=()
TOTAL=${#ARCHS[@]}
IDX=0

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

    OUTDIR="$RESULTS/nsga2_${arch}_s42${SUFFIX}"
    LOGFILE="$LOGS/nsga2_${arch}${SUFFIX}.log"

    if [[ -f "$OUTDIR/pareto_front.json" ]]; then
        log "SKIP $arch — pareto_front.json already exists at $OUTDIR"
        continue
    fi

    log "--- Arch $IDX/$TOTAL: $arch ---"
    log "START $arch | log=$LOGFILE"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY-RUN: python3 $ROOT/scripts/run_nsga2.py --arch $arch --out-suffix $SUFFIX"
        continue
    fi

    set +e
    python3 "$ROOT/scripts/run_nsga2.py" \
        --arch "$arch" \
        --pop-size 50 \
        --generations 30 \
        --seed 42 \
        --out-suffix "$SUFFIX" \
        2>&1 | tee -a "$LOGFILE"
    rc=${PIPESTATUS[0]}
    set -e

    if [[ "$rc" -ne 0 ]]; then
        log "FAILED $arch (exit $rc) — see $LOGFILE"
        FAILED+=("$arch")
        log "Stopping queue."
        break
    fi

    log "DONE $arch"
done

log "============================================================"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    log "All architectures completed successfully."
else
    log "FAILED: ${FAILED[*]}"
    exit 1
fi
log "============================================================"
