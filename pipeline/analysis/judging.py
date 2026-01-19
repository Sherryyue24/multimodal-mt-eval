"""
LLM-as-a-Judge for pairwise comparison of translations.

Compares text-only vs text-image translations using GPT-4.
"""

import time
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import json
import jsonlines

from ..processing.schemas import Prediction, JudgeResult


# Prompt templates for pairwise comparison
JUDGE_PROMPT_TEMPLATE = """You are an expert translation quality evaluator. Compare these two translations and determine which is better.

Source text ({source_lang}):
{source_text}

Translation A:
{translation_a}

Translation B:
{translation_b}

Target language: {target_lang}

Evaluate based on:
1. Accuracy - Does it convey the correct meaning?
2. Fluency - Is it natural in the target language?
3. Cultural appropriateness - Are cultural nuances handled well?

Respond in JSON format:
{{
    "winner": "A" or "B" or "tie",
    "tie_reason": "similar_quality" or "both_bad" or "both_good" or "unclear" (only if tie),
    "confidence": "high" or "medium" or "low",
    "reason": "Brief explanation (1-2 sentences)"
}}"""


class GPTJudge:
    """
    LLM-as-a-Judge using OpenAI GPT models.
    """
    
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self.client = None
    
    def setup(self):
        """Initialize OpenAI client."""
        from openai import OpenAI
        import os
        
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")
        
        self.client = OpenAI(api_key=api_key)
    
    def judge_pair(
        self,
        source_text: str,
        source_lang: str,
        target_lang: str,
        translation_a: str,
        translation_b: str,
        a_mode: str = "text_only",
        b_mode: str = "text_image"
    ) -> Dict[str, Any]:
        """
        Judge a pair of translations.
        
        Args:
            source_text: Original text
            source_lang: Source language
            target_lang: Target language
            translation_a: First translation (typically text_only)
            translation_b: Second translation (typically text_image)
            a_mode: Mode label for A
            b_mode: Mode label for B
            
        Returns:
            Dict with winner, confidence, reason, etc.
        """
        if self.client is None:
            raise RuntimeError("Client not set up. Call setup() first.")
        
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            source_lang=source_lang,
            source_text=source_text,
            translation_a=translation_a,
            translation_b=translation_b,
            target_lang=target_lang
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert translation quality evaluator. Always respond in valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=200
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            # Handle potential markdown code blocks
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            
            result = json.loads(result_text)
            
            # Map A/B winner to actual modes
            winner_raw = result.get("winner", "tie")
            if winner_raw == "A":
                winner = a_mode
            elif winner_raw == "B":
                winner = b_mode
            else:
                winner = "tie"
            
            return {
                "winner": winner,
                "tie_reason": result.get("tie_reason"),
                "confidence": result.get("confidence", "medium"),
                "reason": result.get("reason", ""),
                "error": None
            }
            
        except Exception as e:
            return {
                "winner": "tie",
                "tie_reason": "unclear",
                "confidence": "low",
                "reason": "",
                "error": str(e)
            }


def run_pairwise_judging(
    text_only_predictions: Path,
    text_image_predictions: Path,
    samples_file: Path,
    output_file: Path,
    model: str = "gpt-4",
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run pairwise judging on predictions.
    
    Args:
        text_only_predictions: Path to text_only predictions.jsonl
        text_image_predictions: Path to text_image predictions.jsonl
        samples_file: Path to samples.jsonl
        output_file: Path to output judge_results.jsonl
        model: GPT model to use
        limit: Max comparisons
        
    Returns:
        Stats dict
    """
    from tqdm import tqdm
    from ..processing.build_samples import load_samples
    
    # Load data
    samples_dict = {s.id: s for s in load_samples(samples_file)}
    
    text_only_dict = {}
    with jsonlines.open(text_only_predictions) as reader:
        for p in reader:
            pred = Prediction.from_dict(p)
            text_only_dict[pred.id] = pred
    
    text_image_dict = {}
    with jsonlines.open(text_image_predictions) as reader:
        for p in reader:
            pred = Prediction.from_dict(p)
            text_image_dict[pred.id] = pred
    
    # Find common IDs (samples with both predictions)
    common_ids = set(text_only_dict.keys()) & set(text_image_dict.keys())
    common_ids = [id for id in common_ids if not text_only_dict[id].error and not text_image_dict[id].error]
    
    if limit:
        common_ids = common_ids[:limit]
    
    if not common_ids:
        print("Warning: No common predictions to judge!")
        return {"total": 0, "judged": 0}
    
    # Setup judge
    judge = GPTJudge(model=model)
    judge.setup()
    
    stats = {
        "total": len(common_ids),
        "text_only_wins": 0,
        "text_image_wins": 0,
        "ties": 0,
        "errors": 0
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with jsonlines.open(output_file, mode='w') as writer:
        for sample_id in tqdm(common_ids, desc="Judging"):
            sample = samples_dict[sample_id]
            text_only_pred = text_only_dict[sample_id]
            text_image_pred = text_image_dict[sample_id]
            
            result = judge.judge_pair(
                source_text=sample.source_text,
                source_lang=sample.source_lang,
                target_lang=sample.target_lang,
                translation_a=text_only_pred.prediction,
                translation_b=text_image_pred.prediction,
                a_mode="text_only",
                b_mode="text_image"
            )
            
            judge_result = JudgeResult(
                id=sample_id,
                target_lang=sample.target_lang,
                judge_model=model,
                judge_prompt_template="pairwise_v1",
                winner=result["winner"],
                tie_reason=result.get("tie_reason"),
                confidence=result["confidence"],
                reason=result["reason"],
                error=result.get("error")
            )
            
            writer.write(judge_result.to_dict())
            
            # Update stats
            if result.get("error"):
                stats["errors"] += 1
            elif result["winner"] == "text_only":
                stats["text_only_wins"] += 1
            elif result["winner"] == "text_image":
                stats["text_image_wins"] += 1
            else:
                stats["ties"] += 1
    
    return stats
