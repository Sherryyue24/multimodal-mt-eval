#!/usr/bin/env python3
"""Summarize CometKiwi evaluation results."""
import json
from pathlib import Path
from collections import defaultdict

base_dir = Path(__file__).parent.parent

# Load scores
text_only = []
text_image = []
by_lang = defaultdict(lambda: {"text_only": [], "text_image": []})

with open(base_dir / "artifacts/scores/text_only_scores.jsonl") as f:
    for line in f:
        d = json.loads(line)
        text_only.append(d["score"])
        by_lang[d["target_lang"]]["text_only"].append(d["score"])

with open(base_dir / "artifacts/scores/text_image_scores.jsonl") as f:
    for line in f:
        d = json.loads(line)
        text_image.append(d["score"])
        by_lang[d["target_lang"]]["text_image"].append(d["score"])

print("=" * 60)
print("CometKiwi EVALUATION RESULTS")
print("=" * 60)
print(f"Total samples: {len(text_only)}")
print()
print(f"Text-Only:  mean = {sum(text_only)/len(text_only):.4f}")
print(f"Text-Image: mean = {sum(text_image)/len(text_image):.4f}")
print()
delta = sum(text_image) / len(text_image) - sum(text_only) / len(text_only)
print(f"Delta (Text-Image - Text-Only): {delta:+.4f}")
print()

# Count wins
wins_ti = sum(1 for a, b in zip(text_only, text_image) if b > a)
wins_to = sum(1 for a, b in zip(text_only, text_image) if a > b)
ties = len(text_only) - wins_ti - wins_to

print(f"Text-Image wins: {wins_ti} ({wins_ti/len(text_only)*100:.1f}%)")
print(f"Text-Only wins:  {wins_to} ({wins_to/len(text_only)*100:.1f}%)")
print(f"Ties:            {ties} ({ties/len(text_only)*100:.1f}%)")

# By language
print()
print("-" * 60)
print("By Target Language:")
print("-" * 60)
print(f"{'Language':<12} {'Text-Only':>10} {'Text-Image':>11} {'Delta':>8} {'TI Wins':>8}")
print("-" * 60)

for lang in sorted(by_lang.keys()):
    to_scores = by_lang[lang]["text_only"]
    ti_scores = by_lang[lang]["text_image"]
    to_mean = sum(to_scores) / len(to_scores)
    ti_mean = sum(ti_scores) / len(ti_scores)
    d = ti_mean - to_mean
    wins = sum(1 for a, b in zip(to_scores, ti_scores) if b > a)
    print(f"{lang:<12} {to_mean:>10.4f} {ti_mean:>11.4f} {d:>+8.4f} {wins:>5}/{len(to_scores)}")

print("=" * 60)
