#!/bin/bash
# Double-click this file in Finder to start overnight inference
# Or run: ./start_overnight.command

cd "$(dirname "$0")"
cd ..

echo "=============================================="
echo "STARTING OVERNIGHT INFERENCE"
echo "Time: $(date)"
echo "=============================================="
echo ""
echo "This window will stay open during the entire run."
echo "You can minimize it and go to sleep."
echo "DO NOT close this window!"
echo ""
echo "Progress will be shown below:"
echo "=============================================="

/Users/yue/Documents/code/nlpproject/.venv/bin/python -u scripts/run_batch_inference.py --mode both --batch-size 100 --cool-down 10

echo ""
echo "=============================================="
echo "INFERENCE COMPLETE!"
echo "Time: $(date)"
echo "=============================================="
echo ""
echo "Press any key to close this window..."
read -n 1
