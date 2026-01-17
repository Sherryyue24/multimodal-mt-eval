"""
Example of batch evaluation with data loading.
"""

from multimodal_mt_eval import MultimodalMTEvaluator
from multimodal_mt_eval.data_loader import MultimodalDataLoader


def main():
    # Prepare sample data
    data = [
        {
            "prediction": "A beautiful sunset over the ocean.",
            "reference": "A stunning sunset over the sea.",
        },
        {
            "prediction": "A red car is parked on the street.",
            "reference": "A red vehicle is parked on the road.",
        },
        {
            "prediction": "People are walking in the city.",
            "reference": "People walk through the urban area.",
        },
    ]
    
    # Initialize evaluator
    evaluator = MultimodalMTEvaluator(
        metrics=["bleu", "bert_score"],
        device="cpu"
    )
    
    # Evaluate batch
    print("Evaluating batch of translations...")
    results = evaluator.evaluate_batch(data)
    
    # Print results
    print("\nBatch Evaluation Results:")
    print("-" * 40)
    print(f"Number of examples: {results['num_examples']}")
    print("\nOverall Scores:")
    for metric, score in results['overall'].items():
        print(f"  {metric.upper()}: {score:.4f}")


if __name__ == "__main__":
    main()
