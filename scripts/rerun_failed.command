#!/bin/bash
# 重跑失败样本 - 在 Terminal.app 中运行
cd /Users/yue/Documents/code/nlpproject/multimodal-mt-eval

echo "=============================================="
echo "重跑失败样本 (新解码参数)"
echo "=============================================="
echo "开始时间: $(date)"
echo ""

# 使用正确的 Python 环境
/Users/yue/Documents/code/nlpproject/.venv/bin/python -u scripts/rerun_failed.py \
    --mode both \
    --failed-ids-text-only artifacts/failed_ids_text_only.txt \
    --failed-ids-text-image artifacts/failed_ids_text_image.txt \
    --samples artifacts/samples/samples.jsonl \
    --old-predictions-dir artifacts/predictions/old \
    --output-dir artifacts/predictions \
    2>&1 | tee logs/rerun_failed.log

echo ""
echo "=============================================="
echo "完成时间: $(date)"
echo "=============================================="

# 保持终端打开
read -p "按回车键关闭..."
