"""
Main evaluator for multimodal machine translation systems.
"""

from typing import List, Dict, Any, Optional
import torch
from PIL import Image
import numpy as np


class MultimodalMTEvaluator:
    """
    Evaluator for multimodal machine translation systems.
    
    This class provides methods to evaluate translation quality using both
    text-only and multimodal metrics.
    """
    
    def __init__(self, metrics: Optional[List[str]] = None, device: str = "cpu"):
        """
        Initialize the evaluator.
        
        Args:
            metrics: List of metric names to use for evaluation.
                    Default: ["bleu", "bert_score"]
            device: Device to run models on ("cpu" or "cuda")
        """
        self.device = device
        self.metrics = metrics or ["bleu", "bert_score"]
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize the specified metrics."""
        from .metrics import BLEUMetric, BERTScoreMetric
        
        self.metric_objects = {}
        
        if "bleu" in self.metrics:
            self.metric_objects["bleu"] = BLEUMetric()
        
        if "bert_score" in self.metrics:
            self.metric_objects["bert_score"] = BERTScoreMetric(device=self.device)
    
    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        images: Optional[List[Any]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate translations using specified metrics.
        
        Args:
            predictions: List of predicted translations
            references: List of reference translations
            images: Optional list of images (PIL Images or paths) for multimodal evaluation
        
        Returns:
            Dictionary of metric names and scores
        """
        results = {}
        
        for metric_name, metric_obj in self.metric_objects.items():
            score = metric_obj.compute(predictions, references)
            results[metric_name] = score
        
        return results
    
    def evaluate_batch(
        self,
        data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of translation examples.
        
        Args:
            data: List of dictionaries, each containing:
                  - "prediction": predicted translation
                  - "reference": reference translation
                  - "image": optional image for multimodal evaluation
        
        Returns:
            Dictionary containing:
                - "overall": overall metric scores
                - "per_example": list of per-example scores
        """
        predictions = [d["prediction"] for d in data]
        references = [d["reference"] for d in data]
        images = [d.get("image") for d in data] if any("image" in d for d in data) else None
        
        overall_scores = self.evaluate(predictions, references, images)
        
        return {
            "overall": overall_scores,
            "num_examples": len(data),
        }
