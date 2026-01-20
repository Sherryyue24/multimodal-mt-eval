#!/bin/bash
# Full overnight inference - runs both text_only and text_image
# No intervention needed - just run and sleep!

cd /Users/yue/Documents/code/nlpproject/multimodal-mt-eval
PYTHON="/Users/yue/Documents/code/nlpproject/.venv/bin/python -u"
LOG_DIR=logs
mkdir -p $LOG_DIR

echo "=============================================="
echo "OVERNIGHT FULL INFERENCE"
echo "Started: $(date)"
echo "=============================================="

# Run text_only (all 265 samples)
echo ""
echo "[1/2] Running TEXT_ONLY inference..."
$PYTHON scripts/run_batch_inference.py \
    --mode text_only \
    --batch-size 100 \
    --cool-down 10 \
    2>&1 | tee $LOG_DIR/text_only_$(date +%Y%m%d).log

echo ""
echo "TEXT_ONLY complete at $(date)"
echo "Cooling down 10 minutes before TEXT_IMAGE..."
sleep 600

# Run text_image (all 265 samples)
echo ""
echo "[2/2] Running TEXT_IMAGE inference..."
$PYTHON scripts/run_batch_inference.py \
    --mode text_image \
    --batch-size 100 \
    --cool-down 10 \
    2>&1 | tee $LOG_DIR/text_image_$(date +%Y%m%d).log

echo ""
echo "=============================================="
echo "ALL INFERENCE COMPLETE"
echo "Finished: $(date)"
echo "=============================================="
echo ""
echo "Results:"
echo "  Text-only predictions: $(wc -l < artifacts/predictions/text_only.jsonl) lines"
echo "  Text-image predictions: $(wc -l < artifacts/predictions/text_image.jsonl) lines"
