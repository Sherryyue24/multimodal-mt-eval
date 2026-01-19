"""
Pipeline configuration.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineConfig:
    """Configuration for the evaluation pipeline."""
    
    # Data
    raw_data_path: str = "data/wmt2025_raw/wmt25.jsonl"
    limit: Optional[int] = None
    filter_has_image: bool = False
    
    # Model
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    device: str = "mps"
    
    # Inference
    do_sample: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    
    # Scoring
    comet_model: str = "Unbabel/wmt22-comet-da"
    
    # Judging
    judge_model: str = "gpt-4"
    run_judging: bool = False  # Disabled by default (requires API key)


# Default configurations for different scenarios
DEBUG_CONFIG = PipelineConfig(
    limit=3,
    filter_has_image=True,
    run_judging=False
)

QUICK_TEST_CONFIG = PipelineConfig(
    limit=10,
    filter_has_image=True,
    run_judging=False
)

FULL_RUN_CONFIG = PipelineConfig(
    limit=None,
    filter_has_image=True,  # Only samples with images
    run_judging=True
)
