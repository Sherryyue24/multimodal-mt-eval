"""
Text-only inference engine.

Uses only the source text for translation, ignoring any image.
"""

import time
from typing import Optional
from pathlib import Path

from .base import BaseInference, create_prediction, get_default_config
from .prompt_builder import build_messages
from ..processing.schemas import Sample, Prediction, InferenceConfig


class TextOnlyInference(BaseInference):
    """
    Text-only translation inference.
    
    Uses Qwen2-VL model but without any image input.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct", device: str = "mps"):
        super().__init__(model_name, device)
        self.mode = "text_only"
    
    def load_model(self):
        """Load Qwen2-VL model for text-only inference."""
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        import torch
        
        print(f"Loading model {self.model_name} on {self.device}...")
        
        # Determine dtype based on device (use 'dtype' not deprecated 'torch_dtype')
        if self.device == "mps":
            model_dtype = torch.float16
        else:
            model_dtype = torch.bfloat16
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=model_dtype,
            device_map=self.device
        )
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        print(f"Model loaded successfully on {self.device}")
    
    def run_inference(
        self, 
        sample: Sample, 
        config: Optional[InferenceConfig] = None
    ) -> Prediction:
        """
        Run text-only inference on a sample.
        
        Args:
            sample: Input sample
            config: Inference configuration (optional, will use defaults)
            
        Returns:
            Prediction object
        """
        import torch
        
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if config is None:
            config = get_default_config(sample.source_length)
        
        start_time = time.time()
        error = None
        prediction_text = ""
        
        try:
            # Build messages using unified API (ensures fair A/B comparison)
            context_instruction = sample.meta.get("prompt_instruction")
            messages = build_messages(
                source_text=sample.source_text,
                source_lang=sample.source_lang,
                target_lang=sample.target_lang,
                mode="text_only",
                image=None,
                context_instruction=context_instruction
            )
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize (text only, no images)
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate (top_k=None suppresses warning when do_sample=False)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=config.do_sample,
                    temperature=config.temperature if config.do_sample else None,
                    top_p=config.top_p if config.do_sample else None,
                    top_k=None,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
            
            # Decode (skip input tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            prediction_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
        except Exception as e:
            error = str(e)
            prediction_text = ""
        
        inference_time = time.time() - start_time
        
        return create_prediction(
            sample=sample,
            mode=self.mode,
            model_name=self.model_name,
            prediction_text=prediction_text,
            inference_time=inference_time,
            config=config,
            error=error
        )


def run_text_only_inference(
    samples_file: Path,
    output_file: Path,
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    device: str = "mps",
    limit: Optional[int] = None
) -> dict:
    """
    Run text-only inference on samples file.
    
    Args:
        samples_file: Path to samples.jsonl
        output_file: Path to output predictions.jsonl
        model_name: Model to use
        device: Device to run on
        limit: Max samples to process
        
    Returns:
        Stats dict
    """
    import jsonlines
    from tqdm import tqdm
    from ..processing.build_samples import load_samples
    
    # Load samples
    samples = list(load_samples(samples_file))
    if limit:
        samples = samples[:limit]
    
    # Initialize inference engine
    engine = TextOnlyInference(model_name=model_name, device=device)
    engine.load_model()
    
    stats = {"total": 0, "success": 0, "error": 0}
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with jsonlines.open(output_file, mode='w') as writer:
            for sample in tqdm(samples, desc="Text-only inference"):
                prediction = engine.run_inference(sample)
                writer.write(prediction.to_dict())
                
                stats["total"] += 1
                if prediction.error:
                    stats["error"] += 1
                else:
                    stats["success"] += 1
    finally:
        engine.unload_model()
    
    return stats
