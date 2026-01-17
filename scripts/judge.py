"""
Day 3.3 - LLM-as-a-Judge Evaluation Script
Use LLM to judge which translation is better.
"""

import jsonlines
import pandas as pd
import yaml
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import openai
import os


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


def build_judge_prompt(source_text: str, translation_a: str, translation_b: str) -> str:
    """Build prompt for LLM judge."""
    return f"""You are an expert translation evaluator. Compare the following two German translations of an English source text and determine which one is better.

Source (English): {source_text}

Translation A: {translation_a}

Translation B: {translation_b}

Evaluate based on:
1. Accuracy: How well does it convey the original meaning?
2. Fluency: How natural does it sound in German?
3. Grammar: Are there any grammatical errors?

Output your judgment in the following JSON format:
{{
  "winner": "A" or "B" or "tie",
  "reason": "Brief explanation (1-2 sentences)"
}}

Only output the JSON, nothing else."""


def call_llm_judge(
    prompt: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    api_key: Optional[str] = None
) -> Dict[str, str]:
    """Call LLM API for judgment."""
    
    # Set up OpenAI client
    if api_key:
        openai.api_key = api_key
    elif os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
    else:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
    
    try:
        response = openai.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse JSON response
        import json
        result_json = json.loads(result)
        
        return result_json
        
    except Exception as e:
        print(f"⚠️  LLM API error: {e}")
        return {"winner": "error", "reason": str(e)}


def judge_translations(
    df: pd.DataFrame,
    config: Dict,
    randomize_order: bool = True
) -> pd.DataFrame:
    """Judge all translation pairs."""
    
    judge_config = config['evaluation']['judge']
    results = []
    
    print(f"\n🧑‍⚖️  Judging {len(df)} translation pairs...")
    
    import random
    if randomize_order:
        random.seed(42)  # For reproducibility
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Judging"):
        
        # Randomize A/B order to avoid position bias
        if randomize_order and random.random() < 0.5:
            translation_a = row['hypothesis_multimodal']
            translation_b = row['hypothesis_text_only']
            order_swapped = True
        else:
            translation_a = row['hypothesis_text_only']
            translation_b = row['hypothesis_multimodal']
            order_swapped = False
        
        # Build prompt
        prompt = build_judge_prompt(
            row['source_text'],
            translation_a,
            translation_b
        )
        
        # Call judge
        try:
            judgment = call_llm_judge(
                prompt,
                model_name=judge_config['model_name'],
                temperature=judge_config['temperature'],
                max_tokens=judge_config['max_tokens']
            )
            
            # Map back to actual translations
            winner_raw = judgment['winner']
            if order_swapped:
                if winner_raw == 'A':
                    winner = 'multimodal'
                elif winner_raw == 'B':
                    winner = 'text_only'
                else:
                    winner = winner_raw
            else:
                if winner_raw == 'A':
                    winner = 'text_only'
                elif winner_raw == 'B':
                    winner = 'multimodal'
                else:
                    winner = winner_raw
            
            result = {
                'id': row['id'],
                'winner': winner,
                'reason': judgment.get('reason', ''),
            }
            
        except Exception as e:
            print(f"\n❌ Error judging {row['id']}: {e}")
            result = {
                'id': row['id'],
                'winner': 'error',
                'reason': str(e),
            }
        
        results.append(result)
        
        # Rate limiting (if using API)
        time.sleep(0.5)
    
    return pd.DataFrame(results)


def compute_judge_statistics(df_judge: pd.DataFrame) -> Dict[str, Any]:
    """Compute statistics from judge results."""
    
    total = len(df_judge)
    wins_text_only = (df_judge['winner'] == 'text_only').sum()
    wins_multimodal = (df_judge['winner'] == 'multimodal').sum()
    ties = (df_judge['winner'] == 'tie').sum()
    errors = (df_judge['winner'] == 'error').sum()
    
    stats = {
        'total': total,
        'text_only_wins': wins_text_only,
        'multimodal_wins': wins_multimodal,
        'ties': ties,
        'errors': errors,
        'multimodal_win_rate': wins_multimodal / (total - errors - ties) if (total - errors - ties) > 0 else 0,
    }
    
    return stats


def print_judge_summary(stats: Dict[str, Any]) -> None:
    """Print judge evaluation summary."""
    print("\n" + "=" * 60)
    print("LLM JUDGE EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 Total Judgments: {stats['total']}")
    print(f"\n🏆 Results:")
    print(f"   Text-only wins:   {stats['text_only_wins']} ({stats['text_only_wins']/stats['total']*100:.1f}%)")
    print(f"   Multimodal wins:  {stats['multimodal_wins']} ({stats['multimodal_wins']/stats['total']*100:.1f}%)")
    print(f"   Ties:             {stats['ties']} ({stats['ties']/stats['total']*100:.1f}%)")
    print(f"   Errors:           {stats['errors']}")
    
    print(f"\n📈 Multimodal Win Rate: {stats['multimodal_win_rate']*100:.1f}%")


def analyze_judge_cases(df: pd.DataFrame, df_judge: pd.DataFrame, n: int = 5) -> None:
    """Analyze specific judge cases."""
    
    # Merge with original data
    df_merged = pd.merge(df, df_judge, on='id')
    
    print("\n" + "=" * 60)
    print(f"SAMPLE MULTIMODAL WINS (Top {n})")
    print("=" * 60)
    
    multimodal_wins = df_merged[df_merged['winner'] == 'multimodal'].head(n)
    for idx, row in multimodal_wins.iterrows():
        print(f"\n[{row['id']}]")
        print(f"Source: {row['source_text'][:80]}...")
        print(f"Text-only:  {row['hypothesis_text_only'][:80]}...")
        print(f"Multimodal: {row['hypothesis_multimodal'][:80]}...")
        print(f"Reason: {row['reason']}")


def main():
    """Main execution for Day 3.3 - LLM Judge Evaluation."""
    print("\n" + "=" * 60)
    print("DAY 3.3 - LLM-as-a-Judge Evaluation")
    print("=" * 60 + "\n")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        print("\nYou can also modify the script to use other LLM APIs.")
        return
    
    # Load configuration
    config = load_config()
    
    # Load translations
    print("📖 Loading translations...")
    text_only_path = config['outputs']['text_only']
    multimodal_path = config['outputs']['text_image']
    
    df = load_translations(text_only_path, multimodal_path)
    print(f"Loaded {len(df)} translation pairs")
    
    # Judge translations
    df_judge = judge_translations(df, config, randomize_order=True)
    
    # Save results
    output_path = config['analysis']['judge_results']
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_judge.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to: {output_path}")
    
    # Compute and display statistics
    stats = compute_judge_statistics(df_judge)
    print_judge_summary(stats)
    
    # Analyze cases
    analyze_judge_cases(df, df_judge, n=3)
    
    print("\n" + "=" * 60)
    print("✅ LLM judge evaluation complete!")
    print("=" * 60)
    print("\n🚀 Next: Error analysis (Day 4)")


if __name__ == "__main__":
    main()
