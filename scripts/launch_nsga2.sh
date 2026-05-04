#!/bin/bash
# Launches NSGA-II runs for resnet50 and mobilenet_v2 in separate tmux sessions.
# Run from the project root:
#   bash scripts/launch_nsga2.sh

cd /home/mussa/skin-fairness-ptq

echo "Launching nsga2_resnet50..."
tmux new-session -d -s nsga2_resnet50 \
  "python3 scripts/run_nsga2.py --arch resnet50 --pop-size 50 --generations 30 2>&1 | tee logs/nsga2_resnet50.log"

echo "Launching nsga2_mobilenet..."
tmux new-session -d -s nsga2_mobilenet \
  "python3 scripts/run_nsga2.py --arch mobilenet_v2 --pop-size 50 --generations 30 2>&1 | tee logs/nsga2_mobilenet_v2.log"

echo ""
echo "Both running. Monitor with:"
echo "  tmux attach -t nsga2_resnet50"
echo "  tmux attach -t nsga2_mobilenet"
echo ""
echo "Check progress:"
echo "  tail -f logs/nsga2_resnet50.log"
echo "  tail -f logs/nsga2_mobilenet_v2.log"
