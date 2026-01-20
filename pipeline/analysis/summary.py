"""
Summary generation for evaluation results.

Generates comprehensive reports for:
1. CometKiwi scores
2. LLM-as-a-Judge results  
3. Human-readable report
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, Optional


def calc_stats(scores: list) -> Dict[str, Any]:
    """Calculate basic statistics for a list of scores."""
    if not scores:
        return {"count": 0, "mean": 0, "min": 0, "max": 0}
    return {
        "count": len(scores),
        "mean": round(sum(scores) / len(scores), 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4)
    }


def generate_cometkiwi_summary(
    text_only_scores_file: Path,
    text_image_scores_file: Path
) -> Dict[str, Any]:
    """
    Generate CometKiwi summary from score files.
    
    Args:
        text_only_scores_file: Path to text_only_scores.jsonl
        text_image_scores_file: Path to text_image_scores.jsonl
        
    Returns:
        Summary dict
    """
    text_only_scores = []
    text_image_scores = []
    by_lang = defaultdict(lambda: {"text_only": [], "text_image": []})
    
    # Load text-only scores
    if text_only_scores_file.exists():
        with open(text_only_scores_file) as f:
            for line in f:
                d = json.loads(line)
                text_only_scores.append(d["score"])
                by_lang[d["target_lang"]]["text_only"].append(d["score"])
    
    # Load text-image scores
    if text_image_scores_file.exists():
        with open(text_image_scores_file) as f:
            for line in f:
                d = json.loads(line)
                text_image_scores.append(d["score"])
                by_lang[d["target_lang"]]["text_image"].append(d["score"])
    
    if not text_only_scores or not text_image_scores:
        return {"error": "Missing score files"}
    
    # Calculate pairwise comparison
    ti_wins = sum(1 for a, b in zip(text_only_scores, text_image_scores) if b > a)
    to_wins = sum(1 for a, b in zip(text_only_scores, text_image_scores) if a > b)
    ties = sum(1 for a, b in zip(text_only_scores, text_image_scores) if abs(a - b) < 0.001)
    
    summary = {
        "metric": "CometKiwi (wmt22-cometkiwi-da)",
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(text_only_scores),
        "overall": {
            "text_only": calc_stats(text_only_scores),
            "text_image": calc_stats(text_image_scores),
            "delta": round(
                sum(text_image_scores) / len(text_image_scores) - 
                sum(text_only_scores) / len(text_only_scores), 4
            )
        },
        "pairwise_comparison": {
            "text_image_wins": ti_wins,
            "text_only_wins": to_wins,
            "ties": ties
        },
        "by_language": {}
    }
    
    # Per-language stats
    for lang in sorted(by_lang.keys()):
        to = by_lang[lang]["text_only"]
        ti = by_lang[lang]["text_image"]
        if to and ti:
            summary["by_language"][lang] = {
                "text_only_mean": round(sum(to) / len(to), 4),
                "text_image_mean": round(sum(ti) / len(ti), 4),
                "delta": round(sum(ti) / len(ti) - sum(to) / len(to), 4),
                "text_image_wins": sum(1 for a, b in zip(to, ti) if b > a),
                "count": len(to)
            }
    
    return summary


def generate_judge_summary(judge_results_file: Path) -> Dict[str, Any]:
    """
    Generate LLM-as-a-Judge summary from results file.
    
    Args:
        judge_results_file: Path to judge_results.jsonl
        
    Returns:
        Summary dict
    """
    if not judge_results_file.exists():
        return {"error": "Judge results file not found"}
    
    results = []
    by_lang = defaultdict(lambda: {"text_only": 0, "text_image": 0, "tie": 0})
    confidence_counts = defaultdict(int)
    
    with open(judge_results_file) as f:
        for line in f:
            d = json.loads(line)
            results.append(d)
            by_lang[d["target_lang"]][d["winner"]] += 1
            confidence_counts[d.get("confidence", "unknown")] += 1
    
    if not results:
        return {"error": "No judge results found"}
    
    total = len(results)
    ti_wins = sum(1 for r in results if r["winner"] == "text_image")
    to_wins = sum(1 for r in results if r["winner"] == "text_only")
    ties = sum(1 for r in results if r["winner"] == "tie")
    errors = sum(1 for r in results if r.get("error"))
    
    summary = {
        "metric": "LLM-as-a-Judge (Pairwise)",
        "judge_model": results[0].get("judge_model", "unknown") if results else "unknown",
        "timestamp": datetime.now().isoformat(),
        "total_samples": total,
        "overall": {
            "text_image_wins": ti_wins,
            "text_only_wins": to_wins,
            "ties": ties,
            "errors": errors,
            "text_image_win_rate": round(ti_wins / total * 100, 1) if total > 0 else 0,
            "text_only_win_rate": round(to_wins / total * 100, 1) if total > 0 else 0
        },
        "confidence_distribution": dict(confidence_counts),
        "by_language": {}
    }
    
    # Per-language stats
    for lang in sorted(by_lang.keys()):
        counts = by_lang[lang]
        lang_total = counts["text_only"] + counts["text_image"] + counts["tie"]
        if lang_total > 0:
            summary["by_language"][lang] = {
                "text_image_wins": counts["text_image"],
                "text_only_wins": counts["text_only"],
                "ties": counts["tie"],
                "text_image_win_rate": round(counts["text_image"] / lang_total * 100, 1),
                "count": lang_total
            }
    
    return summary


def generate_text_report(
    comet_summary: Dict[str, Any],
    judge_summary: Dict[str, Any]
) -> str:
    """
    Generate human-readable text report.
    
    Args:
        comet_summary: CometKiwi summary dict
        judge_summary: LLM Judge summary dict
        
    Returns:
        Formatted text report
    """
    lines = []
    
    lines.append("=" * 70)
    lines.append("MULTIMODAL MT EVALUATION REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")
    
    # CometKiwi Section
    lines.append("1. CometKiwi (Reference-Free MT Quality)")
    lines.append("-" * 70)
    
    if "error" not in comet_summary:
        lines.append(f"   Model: wmt22-cometkiwi-da")
        lines.append(f"   Total Samples: {comet_summary['total_samples']}")
        lines.append("")
        lines.append(f"   {'Mode':<15} {'Mean Score':>12} {'Min':>8} {'Max':>8}")
        lines.append(f"   {'-'*43}")
        
        to = comet_summary['overall']['text_only']
        ti = comet_summary['overall']['text_image']
        lines.append(f"   {'Text-Only':<15} {to['mean']:>12.4f} {to['min']:>8.4f} {to['max']:>8.4f}")
        lines.append(f"   {'Text-Image':<15} {ti['mean']:>12.4f} {ti['min']:>8.4f} {ti['max']:>8.4f}")
        lines.append("")
        lines.append(f"   Delta (Text-Image - Text-Only): {comet_summary['overall']['delta']:+.4f}")
        lines.append("")
        
        pw = comet_summary['pairwise_comparison']
        lines.append(f"   Pairwise: Text-Image wins {pw['text_image_wins']}, "
                    f"Text-Only wins {pw['text_only_wins']}, Ties {pw['ties']}")
    else:
        lines.append(f"   Error: {comet_summary['error']}")
    
    lines.append("")
    
    # LLM Judge Section
    lines.append("2. LLM-as-a-Judge (Pairwise Comparison)")
    lines.append("-" * 70)
    
    if "error" not in judge_summary:
        lines.append(f"   Judge Model: {judge_summary['judge_model']}")
        lines.append(f"   Total Samples: {judge_summary['total_samples']}")
        lines.append("")
        lines.append(f"   {'Winner':<20} {'Count':>8} {'Rate':>10}")
        lines.append(f"   {'-'*38}")
        
        ov = judge_summary['overall']
        total = judge_summary['total_samples']
        lines.append(f"   {'Text-Image':<20} {ov['text_image_wins']:>8} {ov['text_image_win_rate']:>9.1f}%")
        lines.append(f"   {'Text-Only':<20} {ov['text_only_wins']:>8} {ov['text_only_win_rate']:>9.1f}%")
        lines.append(f"   {'Tie':<20} {ov['ties']:>8} {ov['ties']/total*100 if total else 0:>9.1f}%")
        lines.append("")
        lines.append(f"   Confidence Distribution: {judge_summary['confidence_distribution']}")
    else:
        lines.append(f"   Error: {judge_summary['error']}")
    
    lines.append("")
    
    # By Language Table
    if "error" not in comet_summary and "error" not in judge_summary:
        lines.append("3. Results by Target Language")
        lines.append("-" * 70)
        lines.append(f"   {'Language':<12} {'CometKiwi':^25} {'LLM Judge':^25}")
        lines.append(f"   {'':<12} {'TO':>8} {'TI':>8} {'Delta':>8} {'TO':>8} {'TI':>8} {'TI%':>8}")
        lines.append(f"   {'-'*66}")
        
        all_langs = set(comet_summary.get('by_language', {}).keys()) | \
                    set(judge_summary.get('by_language', {}).keys())
        
        for lang in sorted(all_langs):
            c = comet_summary.get('by_language', {}).get(lang, {})
            j = judge_summary.get('by_language', {}).get(lang, {})
            lines.append(
                f"   {lang:<12} "
                f"{c.get('text_only_mean', 0):>8.4f} "
                f"{c.get('text_image_mean', 0):>8.4f} "
                f"{c.get('delta', 0):>+8.4f} "
                f"{j.get('text_only_wins', 0):>8} "
                f"{j.get('text_image_wins', 0):>8} "
                f"{j.get('text_image_win_rate', 0):>7.1f}%"
            )
        
        lines.append("")
    
    # Key Findings
    lines.append("=" * 70)
    lines.append("KEY FINDINGS")
    lines.append("=" * 70)
    lines.append("")
    
    if "error" not in comet_summary:
        delta = comet_summary['overall']['delta']
        lines.append(f"✓ Text-Image {'outperforms' if delta > 0 else 'underperforms'} "
                    f"Text-Only on CometKiwi by {delta:+.4f}")
    
    if "error" not in judge_summary:
        rate = judge_summary['overall']['text_image_win_rate']
        lines.append(f"✓ LLM Judge prefers Text-Image in {rate:.1f}% of cases")
    
    lines.append("")
    
    # Top languages
    if "error" not in comet_summary and comet_summary.get('by_language'):
        sorted_langs = sorted(
            comet_summary['by_language'].items(),
            key=lambda x: x[1]['delta'],
            reverse=True
        )
        
        lines.append("Top 3 languages where images help most (CometKiwi delta):")
        for lang, data in sorted_langs[:3]:
            lines.append(f"   {lang}: {data['delta']:+.4f}")
        lines.append("")
        
        lines.append("Top 3 languages where images hurt (CometKiwi delta):")
        for lang, data in sorted_langs[-3:]:
            lines.append(f"   {lang}: {data['delta']:+.4f}")
        lines.append("")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


def generate_full_summary(
    scores_dir: Path,
    output_dir: Path,
    judge_results_file: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Generate all summary files.
    
    Args:
        scores_dir: Directory containing score files
        output_dir: Directory to save summaries
        judge_results_file: Path to judge_results.jsonl (optional, defaults to scores_dir)
        
    Returns:
        Dict with paths to generated files and summary data
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate CometKiwi summary
    comet_summary = generate_cometkiwi_summary(
        text_only_scores_file=scores_dir / "text_only_scores.jsonl",
        text_image_scores_file=scores_dir / "text_image_scores.jsonl"
    )
    
    comet_file = output_dir / "cometkiwi_summary.json"
    with open(comet_file, 'w') as f:
        json.dump(comet_summary, f, indent=2)
    
    # Generate Judge summary
    if judge_results_file is None:
        judge_results_file = scores_dir / "judge_results.jsonl"
    
    judge_summary = generate_judge_summary(judge_results_file)
    
    judge_file = output_dir / "llm_judge_summary.json"
    with open(judge_file, 'w') as f:
        json.dump(judge_summary, f, indent=2)
    
    # Generate text report
    report_text = generate_text_report(comet_summary, judge_summary)
    
    report_file = output_dir / "evaluation_report.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    return {
        "files": {
            "cometkiwi_summary": str(comet_file),
            "llm_judge_summary": str(judge_file),
            "evaluation_report": str(report_file)
        },
        "comet_summary": comet_summary,
        "judge_summary": judge_summary
    }
