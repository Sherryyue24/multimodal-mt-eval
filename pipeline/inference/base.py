"""
Base classes and utilities for inference.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time

from ..processing.schemas import Sample, Prediction, InferenceConfig, get_max_new_tokens


class BaseInference(ABC):
    """Base class for all inference engines."""
    
    def __init__(self, model_name: str, device: str = "mps"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
    
    @abstractmethod
    def load_model(self):
        """Load the model and processor."""
        pass
    
    @abstractmethod
    def run_inference(self, sample: Sample, config: InferenceConfig) -> Prediction:
        """Run inference on a single sample."""
        pass
    
    def batch_inference(self, samples: List[Sample], config: InferenceConfig) -> List[Prediction]:
        """
        Run inference on a batch of samples.
        
        Default implementation is sequential; subclasses may override for batching.
        """
        return [self.run_inference(sample, config) for sample in samples]
    
    def unload_model(self):
        """Unload model to free memory."""
        self.model = None
        self.processor = None
        
        # Force garbage collection
        import gc
        import torch
        gc.collect()
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()


def create_prediction(
    sample: Sample,
    mode: str,
    model_name: str,
    prediction_text: str,
    inference_time: float,
    config: InferenceConfig,
    error: Optional[str] = None
) -> Prediction:
    """Helper to create a Prediction object."""
    return Prediction(
        id=sample.id,
        mode=mode,
        model=model_name,
        prediction=prediction_text,
        target_lang=sample.target_lang,
        inference_time_sec=inference_time,
        config={
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
            "temperature": config.temperature,
            "top_p": config.top_p
        },
        error=error
    )


def get_default_config(source_length: int) -> InferenceConfig:
    """Get default inference config based on source length."""
    return InferenceConfig(
        max_new_tokens=get_max_new_tokens(source_length),
        do_sample=False,
        temperature=None,
        top_p=None
    )
