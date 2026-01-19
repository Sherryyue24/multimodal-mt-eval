#!/usr/bin/env python3
"""Day 0 环境与数据健康检查"""
import json
from pathlib import Path

print('=' * 50)
print('Day 0 健康检查')
print('=' * 50)
print()

# 1. GPU 检查
print('1️⃣ GPU 可用性:')
try:
    import torch
    print(f'   MPS available: {torch.backends.mps.is_available()}')
    print(f'   CUDA available: {torch.cuda.is_available()}')
    if torch.backends.mps.is_available():
        print('   ✅ 推荐使用 MPS (Mac GPU)')
    elif torch.cuda.is_available():
        print(f'   ✅ CUDA: {torch.cuda.get_device_name(0)}')
    else:
        print('   ⚠️  只能用 CPU')
except ImportError:
    print('   ❌ torch 未安装')
print()

# 2. 关键包检查
print('2️⃣ 关键依赖:')
deps = [
    ('transformers', 'Qwen2VLForConditionalGeneration'),
    ('jsonlines', None),
    ('PIL', 'Image'),
    ('tqdm', None),
]
for pkg, attr in deps:
    try:
        m = __import__(pkg)
        if attr:
            getattr(m, attr)
        print(f'   ✅ {pkg}')
    except:
        print(f'   ❌ {pkg}')
print()

# 3. 数据检查
print('3️⃣ 数据完整性:')
raw_path = Path('data/wmt2025_raw/wmt25.jsonl')
if raw_path.exists():
    with open(raw_path) as f:
        lines = f.readlines()
    print(f'   ✅ wmt25.jsonl: {len(lines)} 条')
else:
    print(f'   ❌ wmt25.jsonl 不存在')
print()

# 4. 图片路径检查
print('4️⃣ 图片资产检查 (抽样):')
assets_dir = Path('data/wmt2025_raw/assets')
if assets_dir.exists():
    # 检查前5个有图片的样本
    checked = 0
    missing = 0
    with open(raw_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get('screenshot'):
                img_dir = assets_dir.parent / d['screenshot']
                if img_dir.exists():
                    pngs = list(img_dir.glob('*.png'))
                    if pngs:
                        checked += 1
                    else:
                        missing += 1
                else:
                    missing += 1
                if checked >= 5:
                    break
    print(f'   抽样检查 {checked + missing} 条，存在 {checked} 条')
    if missing == 0:
        print('   ✅ 图片资产正常')
    else:
        print(f'   ⚠️  {missing} 条图片缺失')
else:
    print('   ❌ assets 目录不存在')
print()

# 5. 模型缓存检查
print('5️⃣ 模型缓存 (可选):')
cache_dir = Path.home() / '.cache/huggingface/hub'
qwen_dirs = list(cache_dir.glob('*qwen*2*vl*')) if cache_dir.exists() else []
if qwen_dirs:
    print(f'   ✅ 已缓存 Qwen2-VL 模型')
else:
    print('   ⚠️  Qwen2-VL 未缓存，首次运行需下载 (~4GB)')
print()

print('=' * 50)
print('检查完成！如果全部 ✅，可以开始 Day 1')
print('=' * 50)
