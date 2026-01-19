"""
Core data schemas for the multimodal MT evaluation pipeline.

All pipeline stages communicate through these schemas via JSONL files.
Schema-first design: downstream stages only depend on schema, not implementation.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Literal
from enum import Enum
import json


# =============================================================================
# Enums
# =============================================================================

class InferenceMode(str, Enum):
    TEXT_ONLY = "text_only"
    TEXT_IMAGE = "text_image"


class JudgeWinner(str, Enum):
    TEXT_ONLY = "text_only"
    TEXT_IMAGE = "text_image"
    TIE = "tie"


class TieReason(str, Enum):
    SIMILAR_QUALITY = "similar_quality"
    BOTH_BAD = "both_bad"
    BOTH_GOOD = "both_good"
    UNCLEAR = "unclear"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# 1. Sample Schema (processing → inference)
# =============================================================================

@dataclass
class Sample:
    """
    Standardized sample for inference.
    
    Inference only looks at: source_text, target_lang, image_path
    Everything else is metadata for analysis.
    """
    id: str
    source_lang: str
    target_lang: str
    source_text: str
    source_length: int  # len(source_text), for bucketing analysis
    image_path: Optional[str] = None  # absolute path or relative to project root
    reference_text: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=lambda: {
        "domain": None,
        "source": None
    })
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Sample":
        return cls(**d)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# =============================================================================
# 2. Prediction Schema (inference → analysis)
# =============================================================================

@dataclass
class InferenceConfig:
    """Inference configuration for reproducibility."""
    max_new_tokens: int
    do_sample: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None


@dataclass
class Prediction:
    """
    Model prediction result.
    
    mode is the core variable for evaluation (text_only vs text_image).
    model/config are minimum info for reproducibility.
    """
    id: str
    mode: str  # "text_only" | "text_image"
    model: str
    prediction: str
    target_lang: str  # inherited from sample
    inference_time_sec: float
    config: Dict[str, Any]
    error: Optional[str] = None  # record failure reason
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Prediction":
        return cls(**d)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# =============================================================================
# 3. Score Schema (analysis internal)
# =============================================================================

@dataclass
class Score:
    """
    Automatic metric score for a prediction.
    """
    id: str
    metric: str  # "cometkiwi", "bleu", etc.
    score: Optional[float]  # None if scoring failed
    mode: str  # "text_only" | "text_image"
    target_lang: str  # for per-language grouping
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Score":
        return cls(**d)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# =============================================================================
# 4. Judge Schema (analysis internal)
# =============================================================================

@dataclass
class JudgeResult:
    """
    LLM-as-a-Judge pairwise comparison result.
    """
    id: str
    target_lang: str
    judge_model: str  # "gpt-4", "gpt-3.5-turbo", etc.
    judge_prompt_template: str  # name of the prompt template used
    winner: str  # "text_only" | "text_image" | "tie"
    tie_reason: Optional[str] = None  # "similar_quality" | "both_bad" | "both_good" | "unclear"
    confidence: str = "medium"  # "high" | "medium" | "low"
    reason: str = ""  # brief explanation
    error: Optional[str] = None  # if API call failed
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JudgeResult":
        return cls(**d)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# =============================================================================
# Utility Functions
# =============================================================================

def get_max_new_tokens(source_length: int) -> int:
    """
    Dynamic max_new_tokens based on source text length.
    
    Based on actual data distribution:
    - 63.4% samples > 4000 chars
    - Max: 11526 chars, Avg: 5426 chars
    """
    if source_length < 1000:
        return 512
    elif source_length < 4000:
        return 1024
    else:
        return 2048
