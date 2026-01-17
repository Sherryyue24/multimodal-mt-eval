"""
Utility script for running Day 2 full-scale inference.
"""

import sys
import yaml
from pathlib import Path

# Import inference functions
sys.path.append(str(Path(__file__).parent.parent))
from scripts.infer_text_only import load_model_and_processor, load_samples, run_inference


def load_config(config_path: str = "config/experiment.yaml"):
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Run full-scale inference for both text-only and multimodal."""
    
    print("\n" + "=" * 60)
    print("DAY 2 - FULL-SCALE INFERENCE")
    print("=" * 60)
    print("\n⚠️  WARNING: Configuration is now FROZEN!")
    print("⚠️  Do not modify config/experiment.yaml after this point!\n")
    
    input("Press Enter to continue...")
    
    config = load_config()
    
    # Load model once for both runs
    print("\n📦 Loading model...")
    model, processor = load_model_and_processor(
        config['model']['name'],
        config['model']['device']
    )
    
    # Load full experiment data
    print(f"\n📖 Loading full experiment data...")
    samples = load_samples(config['data']['experiment_data'])
    print(f"Total samples: {len(samples)}")
    
    # Run text-only inference
    print("\n" + "=" * 60)
    print("STEP 1: Text-only Inference")
    print("=" * 60)
    run_inference(
        samples,
        model,
        processor,
        config,
        config['outputs']['text_only']
    )
    
    # Run multimodal inference
    print("\n" + "=" * 60)
    print("STEP 2: Multimodal Inference")
    print("=" * 60)
    
    # Import multimodal inference
    from scripts.infer_multimodal import run_inference as run_multimodal_inference
    
    run_multimodal_inference(
        samples,
        model,
        processor,
        config,
        config['outputs']['text_image']
    )
    
    print("\n" + "=" * 60)
    print("✅ FULL-SCALE INFERENCE COMPLETE!")
    print("=" * 60)
    print(f"\nOutputs saved:")
    print(f"  - Text-only: {config['outputs']['text_only']}")
    print(f"  - Multimodal: {config['outputs']['text_image']}")
    print("\n🚀 Next: Run evaluation scripts (Day 3)")


if __name__ == "__main__":
    main()
