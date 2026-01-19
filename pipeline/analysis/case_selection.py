"""
Case selection for qualitative analysis.

Select interesting examples where text_image performs differently from text_only.
"""

from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import jsonlines

from ..processing.schemas import Sample, Prediction, Score, JudgeResult


@dataclass
class CaseStudy:
    """A case study example for qualitative analysis."""
    id: str
    source_lang: str
    target_lang: str
    source_text: str
    image_path: Optional[str]
    text_only_translation: str
    text_image_translation: str
    text_only_score: Optional[float]
    text_image_score: Optional[float]
    score_delta: Optional[float]
    judge_winner: Optional[str]
    judge_reason: Optional[str]
    category: str  # "text_image_better", "text_only_better", "tie", "error_case"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "source_text": self.source_text[:200] + "..." if len(self.source_text) > 200 else self.source_text,
            "image_path": self.image_path,
            "text_only_translation": self.text_only_translation[:200] + "..." if len(self.text_only_translation) > 200 else self.text_only_translation,
            "text_image_translation": self.text_image_translation[:200] + "..." if len(self.text_image_translation) > 200 else self.text_image_translation,
            "text_only_score": self.text_only_score,
            "text_image_score": self.text_image_score,
            "score_delta": self.score_delta,
            "judge_winner": self.judge_winner,
            "judge_reason": self.judge_reason,
            "category": self.category
        }


def select_case_studies(
    samples_file: Path,
    text_only_predictions: Path,
    text_image_predictions: Path,
    text_only_scores: Optional[Path] = None,
    text_image_scores: Optional[Path] = None,
    judge_results: Optional[Path] = None,
    n_per_category: int = 5
) -> Dict[str, List[CaseStudy]]:
    """
    Select case studies for qualitative analysis.
    
    Categories:
    - text_image_better: Score delta > 0.1 or judge winner is text_image
    - text_only_better: Score delta < -0.1 or judge winner is text_only
    - tie: Similar scores or judge tie
    - error_case: Cases where one mode failed
    
    Args:
        samples_file: Path to samples.jsonl
        text_only_predictions: Path to text_only predictions
        text_image_predictions: Path to text_image predictions
        text_only_scores: Path to text_only scores (optional)
        text_image_scores: Path to text_image scores (optional)
        judge_results: Path to judge results (optional)
        n_per_category: Number of cases per category
        
    Returns:
        Dict mapping category -> list of case studies
    """
    from ..processing.build_samples import load_samples
    
    # Load all data
    samples = {s.id: s for s in load_samples(samples_file)}
    
    text_only_preds = {}
    with jsonlines.open(text_only_predictions) as reader:
        for p in reader:
            pred = Prediction.from_dict(p)
            text_only_preds[pred.id] = pred
    
    text_image_preds = {}
    with jsonlines.open(text_image_predictions) as reader:
        for p in reader:
            pred = Prediction.from_dict(p)
            text_image_preds[pred.id] = pred
    
    # Load scores if available
    text_only_score_dict = {}
    text_image_score_dict = {}
    
    if text_only_scores and text_only_scores.exists():
        with jsonlines.open(text_only_scores) as reader:
            for s in reader:
                score = Score.from_dict(s)
                text_only_score_dict[score.id] = score.score
    
    if text_image_scores and text_image_scores.exists():
        with jsonlines.open(text_image_scores) as reader:
            for s in reader:
                score = Score.from_dict(s)
                text_image_score_dict[score.id] = score.score
    
    # Load judge results if available
    judge_dict = {}
    if judge_results and judge_results.exists():
        with jsonlines.open(judge_results) as reader:
            for j in reader:
                result = JudgeResult.from_dict(j)
                judge_dict[result.id] = result
    
    # Find common IDs
    common_ids = set(text_only_preds.keys()) & set(text_image_preds.keys())
    
    # Categorize cases
    cases: Dict[str, List[CaseStudy]] = {
        "text_image_better": [],
        "text_only_better": [],
        "tie": [],
        "error_case": []
    }
    
    for sample_id in common_ids:
        sample = samples.get(sample_id)
        if not sample:
            continue
        
        to_pred = text_only_preds[sample_id]
        ti_pred = text_image_preds[sample_id]
        
        to_score = text_only_score_dict.get(sample_id)
        ti_score = text_image_score_dict.get(sample_id)
        
        judge = judge_dict.get(sample_id)
        
        # Determine category
        if to_pred.error or ti_pred.error:
            category = "error_case"
        elif judge:
            if judge.winner == "text_image":
                category = "text_image_better"
            elif judge.winner == "text_only":
                category = "text_only_better"
            else:
                category = "tie"
        elif to_score and ti_score:
            delta = ti_score - to_score
            if delta > 0.1:
                category = "text_image_better"
            elif delta < -0.1:
                category = "text_only_better"
            else:
                category = "tie"
        else:
            continue  # Skip if we can't categorize
        
        case = CaseStudy(
            id=sample_id,
            source_lang=sample.source_lang,
            target_lang=sample.target_lang,
            source_text=sample.source_text,
            image_path=sample.image_path,
            text_only_translation=to_pred.prediction,
            text_image_translation=ti_pred.prediction,
            text_only_score=to_score,
            text_image_score=ti_score,
            score_delta=(ti_score - to_score) if (to_score and ti_score) else None,
            judge_winner=judge.winner if judge else None,
            judge_reason=judge.reason if judge else None,
            category=category
        )
        
        cases[category].append(case)
    
    # Sort and limit
    for cat in cases:
        # Sort by score delta (most extreme first)
        if cat == "text_image_better":
            cases[cat].sort(key=lambda c: -(c.score_delta or 0))
        elif cat == "text_only_better":
            cases[cat].sort(key=lambda c: c.score_delta or 0)
        
        cases[cat] = cases[cat][:n_per_category]
    
    return cases


def export_case_studies(
    cases: Dict[str, List[CaseStudy]],
    output_file: Path
):
    """Export case studies to JSON file."""
    import json
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    export = {
        cat: [c.to_dict() for c in case_list]
        for cat, case_list in cases.items()
    }
    
    with open(output_file, 'w') as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
