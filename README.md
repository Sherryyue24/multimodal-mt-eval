# Multimodal Machine Translation Evaluation

A comprehensive Python framework for evaluating multimodal machine translation systems using both text-based and multimodal metrics.

**🧪 Now includes a complete 4-day experimental pipeline for Qwen2-VL-2B multimodal MT evaluation!**

👉 **See [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) for the step-by-step experiment workflow.**

## Features

### Evaluation Framework
- 📊 **Multiple Evaluation Metrics**: Support for BLEU, BERTScore, CometKiwi
- 🖼️ **Multimodal Support**: Designed to evaluate translations with visual context
- 🚀 **Easy to Use**: Simple API for both single and batch evaluation
- 🔧 **Extensible**: Easy to add custom metrics and data loaders
- 🧪 **Well-Tested**: Comprehensive test suite included

### Experimental Pipeline (NEW!)
- 🎯 **4-Day Experiment Protocol**: Complete workflow for multimodal MT experiments
- 🤖 **Qwen2-VL Integration**: Ready-to-use inference scripts for Qwen2-VL-2B
- 📈 **Automated Evaluation**: CometKiwi + LLM-as-a-Judge evaluation
- 🔍 **Error Analysis**: Systematic analysis of when multimodal context helps/hurts
- 💾 **Robust Pipeline**: Incremental saving, crash recovery, reproducible results

## Installation

### From Source

```bash
git clone https://github.com/Sherryyue24/multimodal-mt-eval.git
cd multimodal-mt-eval
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- See `requirements.txt` for full list of dependencies

## Quick Start

### 🚀 Run the 4-Day Experiment

To run the complete Qwen2-VL-2B multimodal MT evaluation experiment:

```bash
# Step 1: Download WMT2025 data (5-10 minutes)
python scripts/prepare_data.py

# Step 2: Day 1 - Text-only vs Multimodal Inference (30-60 minutes)
python scripts/infer_text_only.py     # Run text-only inference
python scripts/infer_multimodal.py    # Run multimodal inference

# Step 3: Day 2 - Full-scale Inference (4-8 hours)
python scripts/run_full_inference.py

# Step 4: Day 3 - Evaluation & Analysis (4-6 hours)
python scripts/eval_comet.py          # Automatic MT quality evaluation
python scripts/judge.py               # LLM-as-Judge pairwise comparison

# Step 5: Day 4 - Error Analysis (manual annotation)
python scripts/analyze_errors.py
```

📋 **See [QUICKSTART.md](QUICKSTART.md) for detailed step-by-step instructions.**

### Basic Usage (Framework API)

```python
from multimodal_mt_eval import MultimodalMTEvaluator

# Sample translations
predictions = [
    "A cat is sitting on a mat.",
    "The dog runs in the park.",
]

references = [
    "A cat is sitting on the mat.",
    "The dog is running in the park.",
]

# Initialize evaluator
evaluator = MultimodalMTEvaluator(
    metrics=["bleu", "bert_score"],
    device="cpu"  # Use "cuda" for GPU
)

# Evaluate
results = evaluator.evaluate(predictions, references)
print(results)
# Output: {'bleu': 75.23, 'bert_score': 0.94}
```

### Batch Evaluation

```python
from multimodal_mt_eval import MultimodalMTEvaluator

data = [
    {
        "prediction": "A beautiful sunset over the ocean.",
        "reference": "A stunning sunset over the sea.",
    },
    {
        "prediction": "A red car is parked on the street.",
        "reference": "A red vehicle is parked on the road.",
    },
]

evaluator = MultimodalMTEvaluator(metrics=["bleu", "bert_score"])
results = evaluator.evaluate_batch(data)
print(results)
```

## Project Structure

```
multimodal-mt-eval/
├── src/
│   └── multimodal_mt_eval/
│       ├── __init__.py
│       ├── evaluator.py      # Main evaluator
│       ├── metrics.py         # Metric implementations
│       └── data_loader.py     # Data loading utilities
├── examples/
│   ├── basic_usage.py         # Basic usage example
│   └── batch_evaluation.py    # Batch evaluation example
├── tests/
│   ├── test_metrics.py
│   └── test_evaluator.py
├── requirements.txt
├── setup.py
└── README.md
```

## Supported Metrics

### Text-based Metrics

- **BLEU**: Bilingual Evaluation Understudy (using SacreBLEU)
- **BERTScore**: Contextual embeddings-based metric for semantic similarity

### Multimodal Metrics (Coming Soon)

- CLIP-based image-text alignment
- Multimodal embedding similarity
- Custom vision-language metrics

## Development

### Running Tests

```bash
# Install development dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=multimodal_mt_eval
```

### Running Examples

```bash
# Basic usage
python examples/basic_usage.py

# Batch evaluation
python examples/batch_evaluation.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{multimodal_mt_eval,
  title = {Multimodal Machine Translation Evaluation Framework},
  author = {Sherryyue24},
  year = {2026},
  url = {https://github.com/Sherryyue24/multimodal-mt-eval}
}
```

## Acknowledgments

- Built with PyTorch and Hugging Face Transformers
- Uses SacreBLEU and BERTScore for evaluation metrics
- Inspired by research in multimodal machine translation

## Contact

For questions or feedback, please open an issue on GitHub.