#!/usr/bin/env python3
"""
Standalone scoring script that can run with Python 3.11.

Supports:
- CometKiwi (reference-free, requires HF login for gated model)
- COMET-DA (needs reference)
- BERTScore (fallback, no login needed)

Usage:
    python3.11 scripts/run_scoring.py --predictions artifacts/predictions/text_only.jsonl
    python3.11 scripts/run_scoring.py --predictions ... --metric bertscore
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import sys

def score_with_comet(predictions, samples_dict, model_name):
    """Score using COMET (CometKiwi or COMET-DA)."""
    from comet import download_model, load_from_checkpoint
    
    print(f"Loading COMET model: {model_name}")
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    
    # Prepare data
    data = []
    for p in predictions:
        sample = samples_dict.get(p['id'])
        if sample:
            data.append({
                "src": sample['source_text'],
                "mt": p['prediction']
            })
    
    print(f"Running COMET on {len(data)} samples...")
    output = model.predict(data, batch_size=8, gpus=0)
    
    return output.scores, output.system_score


def score_with_bertscore(predictions, samples_dict, lang="en"):
    """Score using BERTScore (reference-free mode by comparing to source)."""
    from bert_score import score as bert_score
    
    sources = []
    translations = []
    
    for p in predictions:
        sample = samples_dict.get(p['id'])
        if sample:
            sources.append(sample['source_text'])
            translations.append(p['prediction'])
    
    print(f"Running BERTScore on {len(translations)} samples...")
    # Use multilingual model
    P, R, F1 = bert_score(translations, sources, lang="multilingual", 
                           model_type="bert-base-multilingual-cased",
                           verbose=True)
    
    scores = F1.tolist()
    system_score = sum(scores) / len(scores) if scores else 0
    
    return scores, system_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl")
    parser.add_argument("--samples", required=True, help="Path to samples.jsonl")
    parser.add_argument("--output", required=True, help="Path to output scores.jsonl")
    parser.add_argument("--metric", default="comet", choices=["comet", "bertscore"],
                        help="Scoring metric to use")
    parser.add_argument("--model", default="Unbabel/wmt22-cometkiwi-da", help="COMET model name")
    args = parser.parse_args()
    
    import jsonlines
    
    # Load samples for source texts
    samples_dict = {}
    with jsonlines.open(args.samples) as reader:
        for s in reader:
            samples_dict[s['id']] = s
    
    # Load predictions
    predictions = []
    with jsonlines.open(args.predictions) as reader:
        for p in reader:
            if p.get('prediction') and not p.get('error'):
                predictions.append(p)
    
    if not predictions:
        print("Warning: No valid predictions to score!")
        json.dump({"total": 0, "scored": 0}, sys.stdout)
        sys.exit(0)
    
    print(f"Scoring {len(predictions)} predictions with {args.metric}...")
    
    # Score based on metric
    if args.metric == "comet":
        try:
            scores_list, system_score = score_with_comet(predictions, samples_dict, args.model)
        except Exception as e:
            print(f"COMET failed: {e}")
            print("Falling back to BERTScore...")
            args.metric = "bertscore"
            scores_list, system_score = score_with_bertscore(predictions, samples_dict)
    else:
        scores_list, system_score = score_with_bertscore(predictions, samples_dict)
    
    # Save scores
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with jsonlines.open(output_path, mode='w') as writer:
        for pred, score_val in zip(predictions, scores_list):
            score = {
                "id": pred['id'],
                "metric": args.metric,
                "score": float(score_val),
                "mode": pred.get('mode', 'unknown'),
                "target_lang": pred.get('target_lang', 'unknown')
            }
            writer.write(score)
    
    # Stats
    stats = {
        "total": len(predictions),
        "scored": len(scores_list),
        "avg_score": sum(scores_list) / len(scores_list) if scores_list else 0,
        "system_score": system_score,
        "metric": args.metric
    }
    
    print(f"\nScoring complete:")
    print(f"  Metric: {stats['metric']}")
    print(f"  Scored: {stats['scored']}")
    print(f"  Avg score: {stats['avg_score']:.4f}")
    print(f"  System score: {stats['system_score']:.4f}")
    print(f"  Output: {output_path}")
    
    # Also save stats
    stats_path = output_path.with_suffix('.stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
