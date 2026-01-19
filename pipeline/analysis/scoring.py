"""
Automatic metric scoring for translations.

Supports CometKiwi (reference-free) and optionally BLEU (if reference available).
"""

import time
from typing import List, Dict, Optional, Any
from pathlib import Path
import jsonlines

from ..processing.schemas import Prediction, Score


class CometKiwiScorer:
    """
    Reference-free translation scoring using CometKiwi.
    
    CometKiwi evaluates translation quality without reference translations.
    """
    
    def __init__(self, model_name: str = "Unbabel/wmt22-comet-da"):
        self.model_name = model_name
        self.model = None
    
    def load_model(self):
        """Load CometKiwi model."""
        from comet import download_model, load_from_checkpoint
        
        print(f"Loading CometKiwi model: {self.model_name}")
        model_path = download_model(self.model_name)
        self.model = load_from_checkpoint(model_path)
        print("CometKiwi loaded successfully")
    
    def score(
        self, 
        sources: List[str], 
        translations: List[str],
        references: Optional[List[str]] = None
    ) -> List[float]:
        """
        Score translations.
        
        Args:
            sources: Source texts
            translations: Model translations
            references: Reference translations (optional for CometKiwi)
            
        Returns:
            List of scores
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Prepare data for COMET
        data = []
        for i, (src, mt) in enumerate(zip(sources, translations)):
            item = {"src": src, "mt": mt}
            if references:
                item["ref"] = references[i]
            data.append(item)
        
        # Run scoring
        output = self.model.predict(data, batch_size=8, gpus=0)
        
        return output.scores
    
    def unload_model(self):
        """Unload model to free memory."""
        self.model = None
        import gc
        gc.collect()


def score_predictions(
    predictions_file: Path,
    samples_file: Path,
    output_file: Path,
    metric: str = "cometkiwi"
) -> Dict[str, Any]:
    """
    Score predictions from a predictions file.
    
    Args:
        predictions_file: Path to predictions.jsonl
        samples_file: Path to samples.jsonl (for source text)
        output_file: Path to output scores.jsonl
        metric: Metric to use ("cometkiwi")
        
    Returns:
        Stats dict
    """
    from ..processing.build_samples import load_samples
    
    # Load samples for source texts
    samples_dict = {s.id: s for s in load_samples(samples_file)}
    
    # Load predictions
    predictions = []
    with jsonlines.open(predictions_file) as reader:
        for p in reader:
            predictions.append(Prediction.from_dict(p))
    
    # Filter out error predictions
    valid_predictions = [p for p in predictions if not p.error]
    
    if not valid_predictions:
        print("Warning: No valid predictions to score!")
        return {"total": 0, "scored": 0}
    
    # Prepare data
    sources = [samples_dict[p.id].source_text for p in valid_predictions]
    translations = [p.prediction for p in valid_predictions]
    
    # Score
    scorer = CometKiwiScorer()
    scorer.load_model()
    
    try:
        scores_list = scorer.score(sources, translations)
    finally:
        scorer.unload_model()
    
    # Build Score objects
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with jsonlines.open(output_file, mode='w') as writer:
        for pred, score_val in zip(valid_predictions, scores_list):
            score = Score(
                id=pred.id,
                metric=metric,
                score=score_val,
                mode=pred.mode,
                target_lang=pred.target_lang
            )
            writer.write(score.to_dict())
    
    return {
        "total": len(predictions),
        "scored": len(valid_predictions),
        "skipped_errors": len(predictions) - len(valid_predictions),
        "avg_score": sum(scores_list) / len(scores_list) if scores_list else 0
    }
