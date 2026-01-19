"""
Aggregation and statistical analysis of evaluation results.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
from collections import defaultdict
import json
import jsonlines
from dataclasses import dataclass

from ..processing.schemas import Score, JudgeResult


@dataclass
class AggregatedStats:
    """Aggregated statistics for a group."""
    count: int
    mean: float
    std: float
    min_val: float
    max_val: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "min": round(self.min_val, 4),
            "max": round(self.max_val, 4)
        }


def compute_stats(values: List[float]) -> AggregatedStats:
    """Compute basic statistics for a list of values."""
    import statistics
    
    if not values:
        return AggregatedStats(0, 0.0, 0.0, 0.0, 0.0)
    
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    
    return AggregatedStats(
        count=len(values),
        mean=mean,
        std=std,
        min_val=min(values),
        max_val=max(values)
    )


def aggregate_scores_by_mode(scores_file: Path) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate scores by inference mode.
    
    Args:
        scores_file: Path to scores.jsonl
        
    Returns:
        Dict mapping mode -> stats
    """
    by_mode: Dict[str, List[float]] = defaultdict(list)
    
    with jsonlines.open(scores_file) as reader:
        for record in reader:
            score = Score.from_dict(record)
            if score.score is not None:
                by_mode[score.mode].append(score.score)
    
    return {
        mode: compute_stats(values).to_dict()
        for mode, values in by_mode.items()
    }


def aggregate_scores_by_language(scores_file: Path) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate scores by target language.
    
    Args:
        scores_file: Path to scores.jsonl
        
    Returns:
        Dict mapping (mode, lang) -> stats
    """
    by_lang: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    
    with jsonlines.open(scores_file) as reader:
        for record in reader:
            score = Score.from_dict(record)
            if score.score is not None:
                by_lang[score.target_lang][score.mode].append(score.score)
    
    result = {}
    for lang, modes in by_lang.items():
        result[lang] = {
            mode: compute_stats(values).to_dict()
            for mode, values in modes.items()
        }
    
    return result


def aggregate_judge_results(judge_file: Path) -> Dict[str, Any]:
    """
    Aggregate judge results.
    
    Args:
        judge_file: Path to judge_results.jsonl
        
    Returns:
        Aggregated stats dict
    """
    winners = defaultdict(int)
    tie_reasons = defaultdict(int)
    confidence = defaultdict(int)
    by_lang = defaultdict(lambda: defaultdict(int))
    
    total = 0
    
    with jsonlines.open(judge_file) as reader:
        for record in reader:
            result = JudgeResult.from_dict(record)
            total += 1
            
            winners[result.winner] += 1
            if result.tie_reason:
                tie_reasons[result.tie_reason] += 1
            confidence[result.confidence] += 1
            by_lang[result.target_lang][result.winner] += 1
    
    return {
        "total": total,
        "winners": dict(winners),
        "tie_reasons": dict(tie_reasons),
        "confidence": dict(confidence),
        "by_language": {
            lang: dict(wins) for lang, wins in by_lang.items()
        },
        "win_rates": {
            winner: count / total if total > 0 else 0
            for winner, count in winners.items()
        }
    }


def generate_summary_report(
    scores_by_mode: Dict[str, Dict],
    scores_by_lang: Dict[str, Dict],
    judge_results: Optional[Dict] = None,
    output_file: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive summary report.
    
    Args:
        scores_by_mode: Scores aggregated by mode
        scores_by_lang: Scores aggregated by language
        judge_results: Judge aggregation results
        output_file: Optional path to save report
        
    Returns:
        Complete summary dict
    """
    summary = {
        "scores": {
            "by_mode": scores_by_mode,
            "by_language": scores_by_lang
        }
    }
    
    if judge_results:
        summary["judge"] = judge_results
    
    # Compute mode comparison
    if "text_only" in scores_by_mode and "text_image" in scores_by_mode:
        text_only_mean = scores_by_mode["text_only"].get("mean", 0)
        text_image_mean = scores_by_mode["text_image"].get("mean", 0)
        
        summary["comparison"] = {
            "text_image_vs_text_only_delta": round(text_image_mean - text_only_mean, 4),
            "text_image_better": text_image_mean > text_only_mean
        }
    
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    return summary
