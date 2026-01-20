#!/usr/bin/env python
"""
Sanity check: Does the model actually USE the image?

Test: Same sample, real image vs random noise image
If outputs are identical → image is being ignored (BAD)
If outputs differ → image is being used (GOOD)
"""

import time
import numpy as np
from PIL import Image
from pathlib import Path

# Setup
project_root = Path(__file__).parent
import sys
sys.path.insert(0, str(project_root))

from pipeline.processing.build_samples import load_samples
from pipeline.inference.text_image import TextImageInference
from pipeline.processing.schemas import InferenceConfig, get_max_new_tokens

# Load sample
samples_file = project_root / 'pipeline' / 'artifacts' / 'samples' / 'test_3samples.jsonl'
samples = list(load_samples(samples_file))
sample = samples[0]

print("=" * 60)
print("SANITY CHECK: Does the model use the image?")
print("=" * 60)
print(f"Sample: {sample.id}")
print(f"Source (first 100 chars): {sample.source_text[:100]}...")
print(f"Real image: {sample.image_path}")
print()

# Load model once
engine = TextImageInference(device="mps")
print("Loading model...")
engine.load_model()

config = InferenceConfig(
    max_new_tokens=100,  # Short for speed
    do_sample=False
)

# Test 1: Real image
print("\n--- Test 1: Real image ---")
real_img = Image.open(sample.image_path).convert("RGB")
# Resize as we do in inference
max_size = 1024
if max(real_img.size) > max_size:
    ratio = max_size / max(real_img.size)
    new_size = (int(real_img.size[0] * ratio), int(real_img.size[1] * ratio))
    real_img = real_img.resize(new_size, Image.LANCZOS)
print(f"Real image size: {real_img.size}")

from pipeline.inference.prompt_builder import build_messages

messages_real = build_messages(
    source_text=sample.source_text,
    source_lang=sample.source_lang,
    target_lang=sample.target_lang,
    mode="text_image",
    image=real_img
)

# Run inference with real image (manually to use our resized image)
import torch
text = engine.processor.apply_chat_template(messages_real, tokenize=False, add_generation_prompt=True)
inputs = engine.processor(text=[text], images=[real_img], padding=True, return_tensors="pt").to("mps")

with torch.no_grad():
    output_ids = engine.model.generate(**inputs, max_new_tokens=100, do_sample=False)

output_ids_trimmed = output_ids[:, inputs.input_ids.shape[1]:]
real_output = engine.processor.batch_decode(output_ids_trimmed, skip_special_tokens=True)[0]
print(f"Output: {real_output[:150]}...")

# Test 2: Random noise image (same size)
print("\n--- Test 2: Random noise image ---")
noise_array = np.random.randint(0, 256, (real_img.size[1], real_img.size[0], 3), dtype=np.uint8)
noise_img = Image.fromarray(noise_array)
print(f"Noise image size: {noise_img.size}")

messages_noise = build_messages(
    source_text=sample.source_text,
    source_lang=sample.source_lang,
    target_lang=sample.target_lang,
    mode="text_image",
    image=noise_img
)

text = engine.processor.apply_chat_template(messages_noise, tokenize=False, add_generation_prompt=True)
inputs = engine.processor(text=[text], images=[noise_img], padding=True, return_tensors="pt").to("mps")

with torch.no_grad():
    output_ids = engine.model.generate(**inputs, max_new_tokens=100, do_sample=False)

output_ids_trimmed = output_ids[:, inputs.input_ids.shape[1]:]
noise_output = engine.processor.batch_decode(output_ids_trimmed, skip_special_tokens=True)[0]
print(f"Output: {noise_output[:150]}...")

# Compare
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

if real_output == noise_output:
    print("❌ FAIL: Outputs are IDENTICAL - model may be ignoring the image!")
else:
    print("✓ PASS: Outputs DIFFER - model is using the image")
    
    # Show difference
    print(f"\nReal output length: {len(real_output)}")
    print(f"Noise output length: {len(noise_output)}")
    
    # Character-level diff
    match_count = sum(1 for a, b in zip(real_output, noise_output) if a == b)
    max_len = max(len(real_output), len(noise_output))
    similarity = match_count / max_len * 100
    print(f"Character similarity: {similarity:.1f}%")

engine.unload_model()
