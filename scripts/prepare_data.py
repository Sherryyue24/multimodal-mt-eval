"""
Day 1.1 - Data Preparation Script
Download and prepare WMT2025 multimodal data for experiments.
"""

import json
import jsonlines
import random
from pathlib import Path
from typing import List, Dict, Any
import yaml
from PIL import Image
from tqdm import tqdm
import sys


def load_config(config_path: str = "config/experiment.yaml") -> Dict:
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_wmt2025_data(output_dir: str):
    """
    Download WMT2025 multimodal translation data using download script.
    """
    print("=" * 60)
    print("📥 Downloading WMT2025 Data")
    print("=" * 60)
    
    from download_data import download_wmt2025_data as download_func
    
    success = download_func(output_dir)
    
    if not success:
        print("\n⚠️  Data download failed or skipped.")
        print("    You can manually download from:")
        print("    - Text: https://data.statmt.org/wmt25/general-mt/wmt25.jsonl")
        print("    - Images: https://data.statmt.org/wmt25/general-mt/wmt25_genmt_assets.zip")
        return False
    
    return True


def load_wmt_data(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load WMT2025 data from the downloaded files.
    Handles the official WMT25 format.
    """
    data_dir = Path(data_dir)
    
    samples = []
    
    # Try JSONL format first (WMT25 format)
    jsonl_files = list(data_dir.glob("*.jsonl"))
    
    if jsonl_files:
        jsonl_file = jsonl_files[0]
        print(f"📖 Loading from: {jsonl_file.name}")
        
        with jsonlines.open(jsonl_file) as reader:
            for idx, item in enumerate(reader):
                # WMT25 format typically has:
                # - id, language, text, assets/images
                
                image_path = None
                
                # Handle different possible formats for images
                if 'assets' in item and isinstance(item['assets'], dict):
                    # If images are nested in assets
                    for key, val in item['assets'].items():
                        if isinstance(val, str) and any(val.endswith(ext) for ext in ['.jpg', '.png', '.jpeg']):
                            image_path = val
                            break
                
                if not image_path and 'image' in item:
                    image_path = item.get('image')
                
                if not image_path and 'img' in item:
                    image_path = item.get('img')
                
                # Build sample
                sample = {
                    "id": item.get("id", f"sample_{idx}"),
                    "source_text": item.get("text", item.get("source", "")),
                    "target_lang": item.get("language", item.get("target_lang", "de")),
                    "image_path": image_path,
                    "raw_item": item,  # Keep original for reference
                }
                
                samples.append(sample)
    
    if not samples:
        print("\n⚠️  No JSONL data found. Creating dummy samples for testing...")
        samples = create_dummy_samples(data_dir)
    else:
        print(f"✅ Loaded {len(samples)} samples from JSONL")
    
    return samples


def create_dummy_samples(data_dir: Path, num_samples: int = 5) -> List[Dict[str, Any]]:
    """Create dummy samples for testing when real data is not available."""
    samples = []
    
    dummy_texts = [
        "A cat sitting on a colorful mat.",
        "The sunset over the mountains is beautiful.",
        "Children playing in the park with a ball.",
        "A delicious meal on a wooden table.",
        "A modern building with glass windows.",
    ]
    
    for i in range(num_samples):
        samples.append({
            "id": f"dummy_{i:03d}",
            "source_text": dummy_texts[i % len(dummy_texts)],
            "target_lang": "de",
            "image_path": str(data_dir / "images" / f"dummy_{i:03d}.jpg"),
        })
    
    print(f"✅ Created {len(samples)} dummy samples")
    return samples


def sample_debug_data(
    all_samples: List[Dict[str, Any]],
    num_debug: int,
    seed: int
) -> List[Dict[str, Any]]:
    """Sample a small subset for debugging (Day 1)."""
    random.seed(seed)
    if len(all_samples) <= num_debug:
        return all_samples
    return random.sample(all_samples, num_debug)


def sample_experiment_data(
    all_samples: List[Dict[str, Any]],
    num_samples: int,
    seed: int
) -> List[Dict[str, Any]]:
    """Sample final experiment dataset (Day 2)."""
    random.seed(seed)
    if len(all_samples) <= num_samples:
        return all_samples
    return random.sample(all_samples, num_samples)


def validate_samples(samples: List[Dict[str, Any]]) -> None:
    """Validate that samples are properly formatted and images exist."""
    print("\n🔍 Validating samples...")
    
    valid_count = 0
    issues = []
    
    for idx, sample in enumerate(samples):
        sample_id = sample.get("id", f"sample_{idx}")
        
        # Check required fields
        if not sample.get("source_text"):
            issues.append(f"{sample_id}: Missing source_text")
            continue
        
        # Check image
        image_path = sample.get("image_path")
        if image_path and Path(image_path).exists():
            try:
                Image.open(image_path).verify()
                valid_count += 1
            except Exception as e:
                issues.append(f"{sample_id}: Invalid image - {e}")
        else:
            issues.append(f"{sample_id}: Image not found - {image_path}")
    
    print(f"\n✅ Valid samples: {valid_count}/{len(samples)}")
    if issues:
        print(f"\n⚠️  Issues found:")
        for issue in issues[:10]:  # Show first 10
            print(f"   - {issue}")
        if len(issues) > 10:
            print(f"   ... and {len(issues) - 10} more")


def save_samples(samples: List[Dict[str, Any]], output_path: str) -> None:
    """Save samples to JSONL format."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(samples)
    
    print(f"\n💾 Saved {len(samples)} samples to: {output_path}")


def main():
    """Main execution for Day 1.1 - Data Preparation."""
    print("\n" + "=" * 60)
    print("DAY 1.1 - Data Preparation")
    print("=" * 60 + "\n")
    
    # Load configuration
    config = load_config()
    data_config = config['data']
    
    # Step 1: Download/prepare raw data
    print("\n📥 Step 1: Download WMT2025 data")
    data_dir = data_config['raw_data_dir']
    
    # Check if data already exists
    data_path = Path(data_dir)
    existing_jsonl = list(data_path.glob("*.jsonl"))
    
    if existing_jsonl:
        print(f"✅ Data already exists: {existing_jsonl[0].name}")
        print("   Skipping download.")
    else:
        print(f"Downloading to: {data_dir}")
        success = download_wmt2025_data(data_dir)
        if not success:
            print("\n⚠️  Download was skipped or failed.")
            print("    You can manually download from:")
            print("    - Text: https://data.statmt.org/wmt25/general-mt/wmt25.jsonl")
            print("    - Images: https://data.statmt.org/wmt25/general-mt/wmt25_genmt_assets.zip")
            print("\n    Then run this script again.")
            return
    
    # Step 2: Load data
    print("\n📖 Step 2: Load data")
    all_samples = load_wmt_data(data_dir)
    print(f"Loaded {len(all_samples)} total samples")
    
    if len(all_samples) == 0:
        print("\n❌ No samples loaded. Cannot continue.")
        return
    
    # Step 3: Create debug subset (3-5 samples for Day 1)
    print("\n🔬 Step 3: Create debug subset")
    debug_samples = sample_debug_data(
        all_samples,
        num_debug=5,  # Fixed to 5 for Day 1
        seed=data_config['random_seed']
    )
    save_samples(debug_samples, data_config['debug_samples'])
    validate_samples(debug_samples)
    
    # Step 4: Create full experiment dataset (50 or 100 samples)
    print("\n🎯 Step 4: Create experiment dataset")
    print(f"Sampling {data_config['num_samples']} samples...")
    experiment_samples = sample_experiment_data(
        all_samples,
        num_samples=data_config['num_samples'],
        seed=data_config['random_seed']
    )
    save_samples(experiment_samples, data_config['experiment_data'])
    
    print("\n" + "=" * 60)
    print("✅ Data preparation complete!")
    print("=" * 60)
    print("\n📋 Summary:")
    print(f"   - Debug samples: {len(debug_samples)} → {data_config['debug_samples']}")
    print(f"   - Experiment samples: {len(experiment_samples)} → {data_config['experiment_data']}")
    print("\n🚀 Next: Run inference scripts (Day 1.2 & 1.3)")


if __name__ == "__main__":
    main()
