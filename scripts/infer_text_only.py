"""
Day 1.2 - Text-only Inference Script
Run Qwen2-VL in text-only mode for translation.
"""

import json
import jsonlines
import yaml
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


def load_config(config_path: str = "config/experiment.yaml") -> Dict:
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model_and_processor(model_name: str, device: str):
    """Load Qwen2-VL model and processor."""
    print(f"Loading model: {model_name}")
    
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        processor = AutoProcessor.from_pretrained(model_name)
        
        if device == "cpu":
            model = model.to(device)
        
        print(f"✅ Model loaded on {device}")
        return model, processor
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\n💡 Tip: Make sure you have access to the model:")
        print("   huggingface-cli login")
        raise


def load_samples(data_path: str) -> List[Dict[str, Any]]:
    """Load samples from JSONL file."""
    samples = []
    with jsonlines.open(data_path) as reader:
        for sample in reader:
            samples.append(sample)
    return samples


def build_text_only_prompt(source_text: str, target_lang: str, prompt_template: str) -> str:
    """Build prompt for text-only translation."""
    return f"""{prompt_template}

Source text: {source_text}

Translation to {target_lang.upper()}:"""


def translate_text_only(
    model,
    processor,
    source_text: str,
    target_lang: str,
    prompt_template: str,
    config: Dict
) -> str:
    """Perform text-only translation."""
    
    # Build prompt
    prompt = build_text_only_prompt(source_text, target_lang, prompt_template)
    
    # Prepare inputs
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    
    # Move to device
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config['model']['max_new_tokens'],
            temperature=config['model'].get('temperature', 0.7),
            top_p=config['model'].get('top_p', 0.9),
            do_sample=True,
        )
    
    # Decode
    generated_ids = outputs[0][len(inputs['input_ids'][0]):]
    translation = processor.decode(generated_ids, skip_special_tokens=True)
    
    return translation.strip()


def run_inference(
    samples: List[Dict[str, Any]],
    model,
    processor,
    config: Dict,
    output_path: str
) -> None:
    """Run inference on all samples."""
    
    prompt_template = config['prompts']['text_only']
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process samples
    results = []
    
    print(f"\n🚀 Starting text-only inference on {len(samples)} samples...")
    
    for sample in tqdm(samples, desc="Translating"):
        try:
            translation = translate_text_only(
                model=model,
                processor=processor,
                source_text=sample['source_text'],
                target_lang=sample['target_lang'],
                prompt_template=prompt_template,
                config=config
            )
            
            result = {
                "id": sample['id'],
                "source_text": sample['source_text'],
                "target_lang": sample['target_lang'],
                "hypothesis_text_only": translation,
            }
            results.append(result)
            
            # Save incrementally (防止中途crash)
            with jsonlines.open(output_path, 'a') as writer:
                writer.write(result)
                
        except Exception as e:
            print(f"\n❌ Error processing {sample['id']}: {e}")
            # Save error record
            result = {
                "id": sample['id'],
                "source_text": sample['source_text'],
                "target_lang": sample['target_lang'],
                "hypothesis_text_only": f"ERROR: {str(e)}",
                "error": True
            }
            results.append(result)
    
    print(f"\n✅ Inference complete! Results saved to: {output_path}")
    print(f"   Total: {len(results)} samples")
    print(f"   Errors: {sum(1 for r in results if r.get('error', False))}")


def main():
    """Main execution for Day 1.2 - Text-only Inference."""
    print("\n" + "=" * 60)
    print("DAY 1.2 - Text-only Inference")
    print("=" * 60 + "\n")
    
    # Load configuration
    config = load_config()
    
    # Load model
    model, processor = load_model_and_processor(
        config['model']['name'],
        config['model']['device']
    )
    
    # Load debug samples
    print(f"\n📖 Loading samples from: {config['data']['debug_samples']}")
    samples = load_samples(config['data']['debug_samples'])
    print(f"Loaded {len(samples)} samples")
    
    # Run inference
    output_path = config['outputs']['text_only'].replace('.jsonl', '_debug.jsonl')
    run_inference(samples, model, processor, config, output_path)
    
    print("\n" + "=" * 60)
    print("✅ Text-only inference complete!")
    print("=" * 60)
    print("\n🚀 Next: Run multimodal inference (Day 1.3)")


if __name__ == "__main__":
    main()
