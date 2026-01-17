"""
Day 3.1 - CometKiwi Evaluation Script
Evaluate translations using CometKiwi metric.
"""

import jsonlines
import pandas as pd
import yaml
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from comet import download_model, load_from_checkpoint


def load_config(config_path: str = "config/experiment.yaml") -> Dict:
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_translations(text_only_path: str, multimodal_path: str) -> pd.DataFrame:
    """Load and merge translation results."""
    
    # Load text-only results
    text_only_data = []
    with jsonlines.open(text_only_path) as reader:
        for item in reader:
            text_only_data.append(item)
    
    # Load multimodal results
    multimodal_data = []
    with jsonlines.open(multimodal_path) as reader:
        for item in reader:
            multimodal_data.append(item)
    
    # Merge by ID
    df_text = pd.DataFrame(text_only_data)
    df_multi = pd.DataFrame(multimodal_data)
    
    df = pd.merge(
        df_text[['id', 'source_text', 'hypothesis_text_only']],
        df_multi[['id', 'hypothesis_multimodal']],
        on='id',
        how='inner'
    )
    
    return df


def load_comet_model(model_name: str):
    """Load CometKiwi model."""
    print(f"Loading Comet model: {model_name}")
    
    try:
        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)
        print("✅ Comet model loaded")
        return model
    except Exception as e:
        print(f"❌ Error loading Comet model: {e}")
        raise


def evaluate_with_comet(
    model,
    sources: List[str],
    hypotheses: List[str],
    batch_size: int = 8
) -> List[float]:
    """Evaluate translations with CometKiwi."""
    
    # Prepare data for Comet (QE model doesn't need references)
    data = [
        {"src": src, "mt": hyp}
        for src, hyp in zip(sources, hypotheses)
    ]
    
    # Run evaluation
    print(f"Evaluating {len(data)} translations...")
    output = model.predict(data, batch_size=batch_size, gpus=1)
    
    return output.scores


def compute_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute evaluation statistics."""
    
    stats = {
        "text_only": {
            "mean": df['comet_text_only'].mean(),
            "std": df['comet_text_only'].std(),
            "min": df['comet_text_only'].min(),
            "max": df['comet_text_only'].max(),
        },
        "multimodal": {
            "mean": df['comet_multimodal'].mean(),
            "std": df['comet_multimodal'].std(),
            "min": df['comet_multimodal'].min(),
            "max": df['comet_multimodal'].max(),
        },
        "delta": {
            "mean": df['delta'].mean(),
            "std": df['delta'].std(),
            "positive_rate": (df['delta'] > 0).sum() / len(df),
        },
        "total_samples": len(df),
    }
    
    return stats


def print_summary(stats: Dict[str, Any]) -> None:
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print("COMET EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 Total Samples: {stats['total_samples']}")
    
    print("\n📝 Text-only Translation:")
    print(f"   Mean:  {stats['text_only']['mean']:.4f}")
    print(f"   Std:   {stats['text_only']['std']:.4f}")
    print(f"   Range: [{stats['text_only']['min']:.4f}, {stats['text_only']['max']:.4f}]")
    
    print("\n🖼️  Multimodal Translation:")
    print(f"   Mean:  {stats['multimodal']['mean']:.4f}")
    print(f"   Std:   {stats['multimodal']['std']:.4f}")
    print(f"   Range: [{stats['multimodal']['min']:.4f}, {stats['multimodal']['max']:.4f}]")
    
    print("\n📈 Delta (Multimodal - Text-only):")
    print(f"   Mean:  {stats['delta']['mean']:.4f}")
    print(f"   Std:   {stats['delta']['std']:.4f}")
    print(f"   Win Rate: {stats['delta']['positive_rate']*100:.1f}%")


def analyze_top_cases(df: pd.DataFrame, top_n: int = 10) -> None:
    """Analyze top improvement and degradation cases."""
    
    print("\n" + "=" * 60)
    print(f"TOP {top_n} IMPROVEMENTS (Multimodal > Text-only)")
    print("=" * 60)
    
    top_improvements = df.nlargest(top_n, 'delta')
    for idx, row in top_improvements.iterrows():
        print(f"\n[{row['id']}] Delta: +{row['delta']:.4f}")
        print(f"Source: {row['source_text'][:80]}...")
        print(f"Text-only score:  {row['comet_text_only']:.4f}")
        print(f"Multimodal score: {row['comet_multimodal']:.4f}")
    
    print("\n" + "=" * 60)
    print(f"TOP {top_n} DEGRADATIONS (Text-only > Multimodal)")
    print("=" * 60)
    
    top_degradations = df.nsmallest(top_n, 'delta')
    for idx, row in top_degradations.iterrows():
        print(f"\n[{row['id']}] Delta: {row['delta']:.4f}")
        print(f"Source: {row['source_text'][:80]}...")
        print(f"Text-only score:  {row['comet_text_only']:.4f}")
        print(f"Multimodal score: {row['comet_multimodal']:.4f}")


def main():
    """Main execution for Day 3.1 - Comet Evaluation."""
    print("\n" + "=" * 60)
    print("DAY 3.1 - CometKiwi Evaluation")
    print("=" * 60 + "\n")
    
    # Load configuration
    config = load_config()
    
    # Load translations
    print("📖 Loading translations...")
    text_only_path = config['outputs']['text_only']
    multimodal_path = config['outputs']['text_image']
    
    df = load_translations(text_only_path, multimodal_path)
    print(f"Loaded {len(df)} translation pairs")
    
    # Load Comet model
    comet_model = load_comet_model(config['evaluation']['comet']['model_name'])
    
    # Evaluate text-only translations
    print("\n📊 Evaluating text-only translations...")
    comet_text_only = evaluate_with_comet(
        comet_model,
        df['source_text'].tolist(),
        df['hypothesis_text_only'].tolist(),
        batch_size=config['evaluation']['comet']['batch_size']
    )
    df['comet_text_only'] = comet_text_only
    
    # Evaluate multimodal translations
    print("\n📊 Evaluating multimodal translations...")
    comet_multimodal = evaluate_with_comet(
        comet_model,
        df['source_text'].tolist(),
        df['hypothesis_multimodal'].tolist(),
        batch_size=config['evaluation']['comet']['batch_size']
    )
    df['comet_multimodal'] = comet_multimodal
    
    # Compute delta
    df['delta'] = df['comet_multimodal'] - df['comet_text_only']
    
    # Save results
    output_path = config['analysis']['comet_scores']
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to: {output_path}")
    
    # Compute and display statistics
    stats = compute_statistics(df)
    print_summary(stats)
    
    # Analyze top cases
    analyze_top_cases(df, top_n=5)
    
    print("\n" + "=" * 60)
    print("✅ Comet evaluation complete!")
    print("=" * 60)
    print("\n🚀 Next: LLM-as-a-Judge evaluation (Day 3.3)")


if __name__ == "__main__":
    main()
