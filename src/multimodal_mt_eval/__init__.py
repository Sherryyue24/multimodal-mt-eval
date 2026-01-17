"""
Multimodal Machine Translation Evaluation Framework
"""

__version__ = "0.1.0"

from .evaluator import MultimodalMTEvaluator
from .metrics import BLEUMetric, BERTScoreMetric, MultimodalMetric

__all__ = [
    "MultimodalMTEvaluator",
    "BLEUMetric",
    "BERTScoreMetric",
    "MultimodalMetric",
]
