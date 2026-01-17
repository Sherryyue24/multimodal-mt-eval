"""
Day 4 - Error Analysis Script
Analyze cases where multimodal helps or hurts translation quality.
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Any


def load_config(config_path: str = "config/experiment.yaml") -> Dict:
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_all_results(config: Dict) -> pd.DataFrame:
    """Load and merge all evaluation results."""
    
    # Load Comet scores
    comet_df = pd.read_csv(config['analysis']['comet_scores'])
    
    # Load judge results (if available)
    judge_path = config['analysis']['judge_results']
    if Path(judge_path).exists():
        judge_df = pd.read_csv(judge_path)
        df = pd.merge(comet_df, judge_df, on='id', how='left')
    else:
        df = comet_df
        print("⚠️  Judge results not found, skipping...")
    
    return df


def categorize_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Categorize samples based on Comet delta."""
    
    def categorize(row):
        delta = row['delta']
        threshold_high = 0.05  # Significant improvement
        threshold_low = -0.05  # Significant degradation
        
        if delta > threshold_high:
            return 'significant_improvement'
        elif delta < threshold_low:
            return 'significant_degradation'
        elif delta > 0:
            return 'slight_improvement'
        elif delta < 0:
            return 'slight_degradation'
        else:
            return 'no_change'
    
    df['category'] = df.apply(categorize, axis=1)
    return df


def analyze_by_category(df: pd.DataFrame) -> None:
    """Analyze distribution by category."""
    
    print("\n" + "=" * 60)
    print("CATEGORY DISTRIBUTION")
    print("=" * 60)
    
    category_counts = df['category'].value_counts()
    total = len(df)
    
    for category, count in category_counts.items():
        percentage = count / total * 100
        print(f"{category:25s}: {count:3d} ({percentage:5.1f}%)")


def extract_top_cases(df: pd.DataFrame, n: int = 10) -> Dict[str, pd.DataFrame]:
    """Extract top improvement and degradation cases."""
    
    cases = {
        'top_improvements': df.nlargest(n, 'delta'),
        'top_degradations': df.nsmallest(n, 'delta'),
    }
    
    return cases


def manual_annotation_template(df: pd.DataFrame, output_path: str) -> None:
    """Create a template for manual error annotation."""
    
    # Select cases for annotation
    top_improvements = df.nlargest(10, 'delta')
    top_degradations = df.nsmallest(10, 'delta')
    
    annotation_df = pd.concat([top_improvements, top_degradations])
    
    # Add annotation columns
    annotation_df['error_type'] = ''  # To be filled manually
    annotation_df['notes'] = ''  # To be filled manually
    
    # Select relevant columns
    columns = [
        'id', 'source_text', 
        'hypothesis_text_only', 'hypothesis_multimodal',
        'comet_text_only', 'comet_multimodal', 'delta',
        'error_type', 'notes'
    ]
    
    annotation_df = annotation_df[columns]
    
    # Save
    annotation_df.to_csv(output_path, index=False)
    print(f"\n💾 Annotation template saved to: {output_path}")
    print(f"   Please manually fill in 'error_type' and 'notes' columns")
    print(f"\n   Suggested error types:")
    print("   - visual_disambiguation: Image helped disambiguate meaning")
    print("   - image_misleading: Image led to incorrect translation")
    print("   - image_irrelevant: Image didn't affect translation")
    print("   - over_interpretation: Model over-interpreted visual details")
    print("   - correct_with_image: Better translation with visual context")
    print("   - other: Other types of errors")


def generate_summary(df: pd.DataFrame, config: Dict) -> str:
    """Generate a text summary of findings."""
    
    summary = []
    summary.append("=" * 60)
    summary.append("EXPERIMENT SUMMARY REPORT")
    summary.append("=" * 60)
    summary.append(f"\nExperiment: {config['experiment']['name']}")
    summary.append(f"Date: {config['experiment']['date']}")
    summary.append(f"Model: {config['model']['name']}")
    summary.append(f"Total Samples: {len(df)}")
    
    summary.append("\n" + "-" * 60)
    summary.append("COMET SCORES")
    summary.append("-" * 60)
    summary.append(f"Text-only Mean:   {df['comet_text_only'].mean():.4f}")
    summary.append(f"Multimodal Mean:  {df['comet_multimodal'].mean():.4f}")
    summary.append(f"Delta Mean:       {df['delta'].mean():.4f}")
    summary.append(f"Multimodal Win Rate: {(df['delta'] > 0).sum() / len(df) * 100:.1f}%")
    
    if 'winner' in df.columns:
        summary.append("\n" + "-" * 60)
        summary.append("JUDGE RESULTS")
        summary.append("-" * 60)
        
        valid_judgments = df[df['winner'].isin(['text_only', 'multimodal', 'tie'])]
        if len(valid_judgments) > 0:
            multimodal_wins = (valid_judgments['winner'] == 'multimodal').sum()
            text_wins = (valid_judgments['winner'] == 'text_only').sum()
            ties = (valid_judgments['winner'] == 'tie').sum()
            
            summary.append(f"Multimodal Wins: {multimodal_wins} ({multimodal_wins/len(valid_judgments)*100:.1f}%)")
            summary.append(f"Text-only Wins:  {text_wins} ({text_wins/len(valid_judgments)*100:.1f}%)")
            summary.append(f"Ties:            {ties} ({ties/len(valid_judgments)*100:.1f}%)")
    
    summary.append("\n" + "-" * 60)
    summary.append("CATEGORY DISTRIBUTION")
    summary.append("-" * 60)
    category_counts = df['category'].value_counts()
    for category, count in category_counts.items():
        summary.append(f"{category}: {count} ({count/len(df)*100:.1f}%)")
    
    summary.append("\n" + "-" * 60)
    summary.append("KEY FINDINGS")
    summary.append("-" * 60)
    
    # Auto-generate some insights
    win_rate = (df['delta'] > 0).sum() / len(df)
    if win_rate > 0.6:
        summary.append("✅ Multimodal input shows strong positive effect (>60% win rate)")
    elif win_rate > 0.5:
        summary.append("⚠️  Multimodal input shows slight positive effect (50-60% win rate)")
    else:
        summary.append("❌ Multimodal input does not improve translation quality")
    
    summary.append("\n" + "=" * 60)
    
    return "\n".join(summary)


def main():
    """Main execution for Day 4 - Error Analysis."""
    print("\n" + "=" * 60)
    print("DAY 4 - Error Analysis")
    print("=" * 60 + "\n")
    
    # Load configuration
    config = load_config()
    
    # Load all results
    print("📖 Loading evaluation results...")
    df = load_all_results(config)
    print(f"Loaded {len(df)} samples")
    
    # Categorize samples
    print("\n🔍 Categorizing samples...")
    df = categorize_samples(df)
    analyze_by_category(df)
    
    # Extract top cases
    print("\n📊 Extracting top cases...")
    cases = extract_top_cases(df, n=10)
    
    print(f"\nTop 10 Improvements:")
    for idx, row in cases['top_improvements'].iterrows():
        print(f"  {row['id']}: delta={row['delta']:+.4f}")
    
    print(f"\nTop 10 Degradations:")
    for idx, row in cases['top_degradations'].iterrows():
        print(f"  {row['id']}: delta={row['delta']:+.4f}")
    
    # Save detailed error cases
    error_cases_path = config['analysis']['error_cases']
    Path(error_cases_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Combine top improvements and degradations
    error_cases_df = pd.concat([
        cases['top_improvements'],
        cases['top_degradations']
    ])
    error_cases_df.to_csv(error_cases_path, index=False)
    print(f"\n💾 Error cases saved to: {error_cases_path}")
    
    # Create manual annotation template
    annotation_path = config['analysis']['error_cases'].replace('.csv', '_annotation.csv')
    manual_annotation_template(df, annotation_path)
    
    # Generate summary report
    print("\n📝 Generating summary report...")
    summary = generate_summary(df, config)
    
    summary_path = config['analysis']['summary']
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    print(f"💾 Summary saved to: {summary_path}")
    
    # Print summary
    print("\n" + summary)
    
    print("\n" + "=" * 60)
    print("✅ Error analysis complete!")
    print("=" * 60)
    print("\n📋 Next steps:")
    print("   1. Review error_cases.csv for detailed analysis")
    print("   2. Fill in manual annotations in error_cases_annotation.csv")
    print("   3. Read summary.txt for overall findings")


if __name__ == "__main__":
    main()
