#!/bin/bash
set -u
ROOT=/home/mussa/skin-fairness-ptq
RESULTS=$ROOT/search/results
LOGS=$ROOT/logs
VGG_PARETO=$RESULTS/nsga2_vgg11_s42/pareto_front.json

QUEUE=(shufflenet_v2_x1_0 mobilenet_v2 resnet18 resnet34)

echo "[$(date)] queue: waiting for $VGG_PARETO"
until [ -f "$VGG_PARETO" ]; do
    sleep 300
done
echo "[$(date)] vgg11 done — starting queue"

for arch in "${QUEUE[@]}"; do
    LOG=$LOGS/nsga2_${arch}_queued.log
    echo "[$(date)] >>> starting nsga2 for $arch (log: $LOG)"
    python3 $ROOT/scripts/run_nsga2.py --arch "$arch" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    echo "[$(date)] <<< finished $arch (rc=$rc)"
done
echo "[$(date)] queue: all done"
