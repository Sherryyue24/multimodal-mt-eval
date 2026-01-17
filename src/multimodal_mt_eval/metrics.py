"""
Evaluation metrics for machine translation.
"""

from typing import List, Union
import numpy as np
from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """Base class for all metrics."""
    
    @abstractmethod
    def compute(self, predictions: List[str], references: List[str]) -> float:
        """Compute metric score."""
        pass


class BLEUMetric(BaseMetric):
    """BLEU score metric using SacreBLEU."""
    
    def __init__(self):
        try:
            import sacrebleu
            self.sacrebleu = sacrebleu
        except ImportError:
            raise ImportError("Please install sacrebleu: pip install sacrebleu")
    
    def compute(self, predictions: List[str], references: List[str]) -> float:
        """
        Compute BLEU score.
        
        Args:
            predictions: List of predicted translations
            references: List of reference translations
        
        Returns:
            BLEU score (0-100)
        """
        # SacreBLEU expects references as list of lists
        refs = [[ref] for ref in references]
        
        # Transpose references for corpus_bleu
        refs_transposed = list(zip(*refs))
        
        bleu = self.sacrebleu.corpus_bleu(predictions, refs_transposed)
        return bleu.score


class BERTScoreMetric(BaseMetric):
    """BERTScore metric for semantic similarity."""
    
    def __init__(self, device: str = "cpu", lang: str = "en"):
        """
        Initialize BERTScore metric.
        
        Args:
            device: Device to run model on
            lang: Language code (e.g., "en", "zh", "de")
        """
        self.device = device
        self.lang = lang
        
        try:
            from bert_score import score
            self.score_fn = score
        except ImportError:
            raise ImportError("Please install bert-score: pip install bert-score")
    
    def compute(self, predictions: List[str], references: List[str]) -> float:
        """
        Compute BERTScore.
        
        Args:
            predictions: List of predicted translations
            references: List of reference translations
        
        Returns:
            F1 BERTScore (0-1)
        """
        P, R, F1 = self.score_fn(
            predictions,
            references,
            lang=self.lang,
            device=self.device,
            verbose=False,
        )
        return F1.mean().item()


class MultimodalMetric(BaseMetric):
    """
    Multimodal metric that considers both text and image information.
    
    This is a placeholder for future implementation of metrics that
    evaluate how well translations capture image-text alignment.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
    
    def compute(self, predictions: List[str], references: List[str]) -> float:
        """
        Compute multimodal metric score.
        
        Note: This is a placeholder implementation.
        """
        # TODO: Implement actual multimodal metric
        # This could involve:
        # - Vision-language models (e.g., CLIP)
        # - Image-text alignment scores
        # - Multimodal embedding similarity
        
        raise NotImplementedError(
            "Multimodal metric is not yet implemented. "
            "This requires integration with vision-language models."
        )
