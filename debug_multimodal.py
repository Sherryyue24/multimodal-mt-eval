#!/usr/bin/env python
"""Minimal test of Qwen2-VL multimodal inference."""

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Load model
print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,
    device_map="mps"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
print("Model loaded!")

# Load image
img_path = "/Users/yue/Documents/code/nlpproject/multimodal-mt-eval/data/wmt2025_raw/assets/en/social/112502991207286008-anon/112502991207286008-anon_2.png"
image = Image.open(img_path).convert("RGB")
print(f"Image: {image.size}")

# Resize if too large (Qwen2-VL might have issues with very large images)
max_size = 1024
if max(image.size) > max_size:
    ratio = max_size / max(image.size)
    new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
    image = image.resize(new_size, Image.LANCZOS)
    print(f"Resized to: {image.size}")

# Build messages - proper Qwen2-VL format
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What is in this image? Describe briefly."}
        ]
    }
]

# Apply chat template
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"\nGenerated prompt (first 500 chars):\n{text[:500]}...")

# Process
inputs = processor(
    text=[text],
    images=[image],
    padding=True,
    return_tensors="pt"
).to("mps")

print(f"\nInput shapes: {inputs.input_ids.shape}")

# Generate
print("\nGenerating...")
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

# Decode
output_ids_trimmed = output_ids[:, inputs.input_ids.shape[1]:]
response = processor.batch_decode(output_ids_trimmed, skip_special_tokens=True)[0]

print(f"\n=== Response ===")
print(response)
