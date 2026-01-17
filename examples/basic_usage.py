"""
Basic usage example for multimodal MT evaluation.
"""

from multimodal_mt_eval import MultimodalMTEvaluator


def main():
    # Sample data
    predictions = [
        "A cat is sitting on a mat.",
        "The dog runs in the park.",
        "Children are playing with toys.",
    ]
    
    references = [
        "A cat is sitting on the mat.",
        "The dog is running in the park.",
        "Children play with toys.",
    ]
    
    # Initialize evaluator
    evaluator = MultimodalMTEvaluator(
        metrics=["bleu", "bert_score"],
        device="cpu"  # Change to "cuda" if GPU is available
    )
    
    # Evaluate
    print("Evaluating translations...")
    results = evaluator.evaluate(predictions, references)
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 40)
    for metric, score in results.items():
        print(f"{metric.upper()}: {score:.4f}")


if __name__ == "__main__":
    main()
