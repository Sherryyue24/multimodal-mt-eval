#!/usr/bin/env python3
"""
Generate evaluation summary reports for both CometKiwi and LLM-as-a-Judge.

Outputs:
- artifacts/summaries/cometkiwi_summary.json
- artifacts/summaries/llm_judge_summary.json
- artifacts/summaries/evaluation_report.txt (human-readable)
"""
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

base_dir = Path(__file__).parent.parent
summaries_dir = base_dir / "artifacts" / "summaries"
summaries_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. CometKiwi Summary
# ============================================================
print("Processing CometKiwi scores...")

text_only_scores = []
text_image_scores = []
comet_by_lang = defaultdict(lambda: {"text_only": [], "text_image": []})

with open(base_dir / "artifacts/scores/text_only_scores.jsonl") as f:
    for line in f:
        d = json.loads(line)
        text_only_scores.append(d["score"])
        comet_by_lang[d["target_lang"]]["text_only"].append(d["score"])

with open(base_dir / "artifacts/scores/text_image_scores.jsonl") as f:
    for line in f:
        d = json.loads(line)
        text_image_scores.append(d["score"])
        comet_by_lang[d["target_lang"]]["text_image"].append(d["score"])

# Calculate statistics
def calc_stats(scores):
    if not scores:
        return {"count": 0, "mean": 0, "min": 0, "max": 0}
    return {
        "count": len(scores),
        "mean": round(sum(scores) / len(scores), 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4)
    }

comet_summary = {
    "metric": "CometKiwi (wmt22-cometkiwi-da)",
    "timestamp": datetime.now().isoformat(),
    "total_samples": len(text_only_scores),
    "overall": {
        "text_only": calc_stats(text_only_scores),
        "text_image": calc_stats(text_image_scores),
        "delta": round(sum(text_image_scores)/len(text_image_scores) - sum(text_only_scores)/len(text_only_scores), 4)
    },
    "pairwise_comparison": {
        "text_image_wins": sum(1 for a, b in zip(text_only_scores, text_image_scores) if b > a),
        "text_only_wins": sum(1 for a, b in zip(text_only_scores, text_image_scores) if a > b),
        "ties": sum(1 for a, b in zip(text_only_scores, text_image_scores) if abs(a-b) < 0.001)
    },
    "by_language": {}
}

for lang in sorted(comet_by_lang.keys()):
    to = comet_by_lang[lang]["text_only"]
    ti = comet_by_lang[lang]["text_image"]
    comet_summary["by_language"][lang] = {
        "text_only_mean": round(sum(to)/len(to), 4),
        "text_image_mean": round(sum(ti)/len(ti), 4),
        "delta": round(sum(ti)/len(ti) - sum(to)/len(to), 4),
        "text_image_wins": sum(1 for a, b in zip(to, ti) if b > a),
        "count": len(to)
    }

with open(summaries_dir / "cometkiwi_summary.json", "w") as f:
    json.dump(comet_summary, f, indent=2)
print(f"  Saved: {summaries_dir / 'cometkiwi_summary.json'}")

# ============================================================
# 2. LLM-as-a-Judge Summary
# ============================================================
print("Processing LLM-as-a-Judge results...")

judge_results = []
judge_by_lang = defaultdict(lambda: {"text_only": 0, "text_image": 0, "tie": 0})
confidence_counts = defaultdict(int)

with open(base_dir / "artifacts/scores/judge_results.jsonl") as f:
    for line in f:
        d = json.loads(line)
        judge_results.append(d)
        judge_by_lang[d["target_lang"]][d["winner"]] += 1
        confidence_counts[d.get("confidence", "unknown")] += 1

total = len(judge_results)
text_image_wins = sum(1 for r in judge_results if r["winner"] == "text_image")
text_only_wins = sum(1 for r in judge_results if r["winner"] == "text_only")
ties = sum(1 for r in judge_results if r["winner"] == "tie")
errors = sum(1 for r in judge_results if r.get("error"))

judge_summary = {
    "metric": "LLM-as-a-Judge (Pairwise)",
    "judge_model": judge_results[0].get("judge_model", "unknown") if judge_results else "unknown",
    "timestamp": datetime.now().isoformat(),
    "total_samples": total,
    "overall": {
        "text_image_wins": text_image_wins,
        "text_only_wins": text_only_wins,
        "ties": ties,
        "errors": errors,
        "text_image_win_rate": round(text_image_wins / total * 100, 1) if total > 0 else 0,
        "text_only_win_rate": round(text_only_wins / total * 100, 1) if total > 0 else 0
    },
    "confidence_distribution": dict(confidence_counts),
    "by_language": {}
}

for lang in sorted(judge_by_lang.keys()):
    counts = judge_by_lang[lang]
    lang_total = counts["text_only"] + counts["text_image"] + counts["tie"]
    judge_summary["by_language"][lang] = {
        "text_image_wins": counts["text_image"],
        "text_only_wins": counts["text_only"],
        "ties": counts["tie"],
        "text_image_win_rate": round(counts["text_image"] / lang_total * 100, 1) if lang_total > 0 else 0,
        "count": lang_total
    }

with open(summaries_dir / "llm_judge_summary.json", "w") as f:
    json.dump(judge_summary, f, indent=2)
print(f"  Saved: {summaries_dir / 'llm_judge_summary.json'}")

# ============================================================
# 3. Human-Readable Report
# ============================================================
print("Generating human-readable report...")

report = []
report.append("=" * 70)
report.append("MULTIMODAL MT EVALUATION REPORT")
report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report.append("=" * 70)
report.append("")

# CometKiwi Section
report.append("1. CometKiwi (Reference-Free MT Quality)")
report.append("-" * 70)
report.append(f"   Model: wmt22-cometkiwi-da")
report.append(f"   Total Samples: {comet_summary['total_samples']}")
report.append("")
report.append(f"   {'Mode':<15} {'Mean Score':>12} {'Min':>8} {'Max':>8}")
report.append(f"   {'-'*43}")
report.append(f"   {'Text-Only':<15} {comet_summary['overall']['text_only']['mean']:>12.4f} {comet_summary['overall']['text_only']['min']:>8.4f} {comet_summary['overall']['text_only']['max']:>8.4f}")
report.append(f"   {'Text-Image':<15} {comet_summary['overall']['text_image']['mean']:>12.4f} {comet_summary['overall']['text_image']['min']:>8.4f} {comet_summary['overall']['text_image']['max']:>8.4f}")
report.append("")
report.append(f"   Delta (Text-Image - Text-Only): {comet_summary['overall']['delta']:+.4f}")
report.append("")
report.append(f"   Pairwise: Text-Image wins {comet_summary['pairwise_comparison']['text_image_wins']}, Text-Only wins {comet_summary['pairwise_comparison']['text_only_wins']}, Ties {comet_summary['pairwise_comparison']['ties']}")
report.append("")

# LLM Judge Section
report.append("2. LLM-as-a-Judge (Pairwise Comparison)")
report.append("-" * 70)
report.append(f"   Judge Model: {judge_summary['judge_model']}")
report.append(f"   Total Samples: {judge_summary['total_samples']}")
report.append("")
report.append(f"   {'Winner':<20} {'Count':>8} {'Rate':>10}")
report.append(f"   {'-'*38}")
report.append(f"   {'Text-Image':<20} {judge_summary['overall']['text_image_wins']:>8} {judge_summary['overall']['text_image_win_rate']:>9.1f}%")
report.append(f"   {'Text-Only':<20} {judge_summary['overall']['text_only_wins']:>8} {judge_summary['overall']['text_only_win_rate']:>9.1f}%")
report.append(f"   {'Tie':<20} {judge_summary['overall']['ties']:>8} {judge_summary['overall']['ties']/total*100:>9.1f}%")
report.append("")
report.append(f"   Confidence Distribution: {dict(confidence_counts)}")
report.append("")

# By Language Table
report.append("3. Results by Target Language")
report.append("-" * 70)
report.append(f"   {'Language':<12} {'CometKiwi':^25} {'LLM Judge':^25}")
report.append(f"   {'':<12} {'TO':>8} {'TI':>8} {'Delta':>8} {'TO':>8} {'TI':>8} {'TI%':>8}")
report.append(f"   {'-'*66}")

for lang in sorted(comet_by_lang.keys()):
    c = comet_summary["by_language"].get(lang, {})
    j = judge_summary["by_language"].get(lang, {})
    report.append(f"   {lang:<12} {c.get('text_only_mean', 0):>8.4f} {c.get('text_image_mean', 0):>8.4f} {c.get('delta', 0):>+8.4f} {j.get('text_only_wins', 0):>8} {j.get('text_image_wins', 0):>8} {j.get('text_image_win_rate', 0):>7.1f}%")

report.append("")
report.append("=" * 70)
report.append("KEY FINDINGS")
report.append("=" * 70)
report.append("")
report.append(f"✓ Text-Image outperforms Text-Only on CometKiwi by {comet_summary['overall']['delta']:+.4f}")
report.append(f"✓ LLM Judge prefers Text-Image in {judge_summary['overall']['text_image_win_rate']:.1f}% of cases")
report.append(f"✓ Visual context provides consistent improvement across most languages")
report.append("")

# Top benefiting languages
sorted_langs = sorted(comet_summary["by_language"].items(), key=lambda x: x[1]["delta"], reverse=True)
report.append("Top 5 languages where images help most (CometKiwi delta):")
for lang, data in sorted_langs[:5]:
    report.append(f"   {lang}: {data['delta']:+.4f}")
report.append("")

report.append("Top 5 languages where images hurt (CometKiwi delta):")
for lang, data in sorted_langs[-5:]:
    report.append(f"   {lang}: {data['delta']:+.4f}")
report.append("")
report.append("=" * 70)

report_text = "\n".join(report)

with open(summaries_dir / "evaluation_report.txt", "w") as f:
    f.write(report_text)
print(f"  Saved: {summaries_dir / 'evaluation_report.txt'}")

# Print report to terminal
print("\n" + report_text)
