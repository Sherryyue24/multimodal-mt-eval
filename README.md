# Multimodal Machine Translation Evaluation

An evaluation pipeline for multimodal machine translation using Large Vision-Language Models (LVLMs). Compares translation quality between text-only and text+image inputs.

**📊 Research Focus**: Does visual context improve translation quality for WMT2025 multimodal data using Qwen2-VL-2B?

## Key Results (265 samples, 28 languages)

| Metric | Text-Only | Text-Image | Δ |
|--------|-----------|------------|---|
| CometKiwi | 0.4628 | 0.4795 | +0.0167 |
| LLM Judge Win Rate | 34.0% | 63.8% | +29.8% |
| Error Rate | 11.3% | 5.3% | -6.0% |

**Conclusion**: Visual context provides modest but consistent improvements, primarily through reduced decoding errors.

## Installation

### Prerequisites
- Python 3.11+ (required for COMET compatibility)
- PyTorch 2.0.0+
- CUDA 11.8+ (for GPU inference) or MPS (Apple Silicon)
- ~15GB free disk space (WMT2025 data + models)

### From Source
```bash
git clone https://github.com/Sherryyue24/multimodal-mt-eval.git
cd multimodal-mt-eval
pip install -e .
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Verify Installation
```bash
python -c "from pipeline.run import run_pipeline; print('✓ Pipeline ready')"
```

## Quick Start

### Run the Complete Pipeline

Execute the full evaluation workflow with a single command:

```bash
# Run complete pipeline with 10 samples (debug mode)
python -m pipeline.run --stage all --limit 10 --device mps

# Run on GPU (recommended for production)
python -m pipeline.run --stage all --device cuda

# Run specific stage only
python -m pipeline.run --stage inference --limit 100
python -m pipeline.run --stage analysis
```

### Pipeline Stages

The evaluation follows a 4-stage architecture:

#### 1. **Processing** - Data Standardization
Converts WMT2025 raw data into unified Sample schema with validation.

```
Input:  data/wmt2025_raw/wmt25.jsonl
Output: artifacts/samples/samples.jsonl
```

#### 2. **Inference** - Dual Prediction Modes  
Runs both text-only and text+image inference using Qwen2-VL-2B-Instruct.

```
Input:  artifacts/samples/samples.jsonl
Output: artifacts/predictions/{text_only,text_image}.jsonl
Config: max_new_tokens dynamically set based on source length
```

#### 3. **Analysis** - Scoring & Judging
Evaluates predictions using CometKiwi (reference-free) and LLM-as-Judge (GPT-4o pairwise).

```
Input:  artifacts/predictions/*.jsonl
Output: artifacts/scores/text_only_scores.jsonl
        artifacts/scores/text_image_scores.jsonl
        artifacts/scores/judge_results.jsonl
```

#### 4. **Summary** - Reports & Error Analysis
Generates comprehensive evaluation reports and error taxonomy analysis.

```
Input:  artifacts/scores/*.jsonl
Output: artifacts/summaries/evaluation_report.txt
        artifacts/summaries/cometkiwi_summary.json
        artifacts/summaries/llm_judge_summary.json
        artifacts/reports/error_taxonomy_report.txt
```

### Example: Check Outputs

```bash
# View sample outputs
cat artifacts/samples/samples.jsonl | head -1 | python -m json.tool
cat artifacts/predictions/text_only.jsonl | head -1 | python -m json.tool
cat artifacts/summaries/evaluation_report.txt
```

### Key Configuration Options

```python
# In pipeline/config.py
# Pipeline stages can be run via CLI:
python -m pipeline.run --stage all      # Run complete pipeline
python -m pipeline.run --stage summary  # Generate reports only

# Environment variables for LLM Judge (.env file):
API_BASE_URL=https://api.openai.com/v1  # Or custom endpoint
API_KEY=sk-xxx
MODEL_NAME=gpt-4o
```

## Project Structure

```
multimodal-mt-eval/
├── pipeline/                          # Main evaluation pipeline (stage-decoupled)
│   ├── run.py                         # Pipeline orchestrator & entry point
│   ├── config.py                      # Centralized configuration
│   │
│   ├── processing/                    # Stage 1: Data Standardization
│   │   ├── schemas.py                 # Unified Sample/Prediction/Score/Judge schemas
│   │   ├── build_samples.py           # WMT2025 → Sample conversion
│   │   └── validators.py              # Schema validation & path existence checks
│   │
│   ├── inference/                     # Stage 2: Model Inference
│   │   ├── base.py                    # InferenceEngine interface
│   │   ├── prompt_builder.py          # Multimodal prompt construction
│   │   ├── text_only.py               # Text-only inference implementation
│   │   └── text_image.py              # Text+Image inference implementation
│   │
│   └── analysis/                      # Stage 3&4: Evaluation & Aggregation
│       ├── scoring.py                 # CometKiwi scoring
│       ├── judging.py                 # LLM-as-Judge pairwise comparison
│       ├── aggregation.py             # Per-language/domain statistics
│       ├── summary.py                 # Report generation
│       ├── error_taxonomy.py          # Error analysis & categorization
│       └── case_selection.py          # Representative case sampling
│
├── data/
│   └── wmt2025_raw/                   # WMT2025 dataset location
│       └── wmt25.jsonl                # Raw multilingual data (~265 samples with images)
│       └── assets/                    # Image files referenced in dataset
│
├── artifacts/                         # Pipeline outputs (versioned & reproducible)
│   ├── samples/                       # Standardized samples from Stage 1
│   │   └── samples.jsonl
│   ├── predictions/                   # Stage 2 outputs
│   │   ├── text_only.jsonl
│   │   └── text_image.jsonl
│   ├── scores/                        # Stage 3 outputs
│   │   ├── text_only_scores.jsonl
│   │   ├── text_image_scores.jsonl
│   │   └── judge_results.jsonl
│   ├── summaries/                     # Stage 5 outputs
│   │   ├── evaluation_report.txt
│   │   ├── cometkiwi_summary.json
│   │   └── llm_judge_summary.json
│   └── reports/                       # Error analysis
│       ├── error_taxonomy_report.txt
│       └── error_taxonomy_data.json
│
├── tests/                             # Test suite
│   ├── test_metrics.py
│   ├── test_evaluator.py
│   └── debug_multimodal.py
│
├── setup.py
├── requirements.txt
└── README.md
```

## Data & Evaluation Details

### Dataset: WMT2025 Multimodal MT
- **Size**: 265 samples with associated screenshots
- **Domain**: Social media content
- **Language Pairs**: EN → 28 target languages
- **Visual Content**: Each sample has an associated screenshot image

### Evaluation Metrics

#### Automatic Metrics
- **CometKiwi** (Primary): Unbabel's reference-free multilingual metric (wmt22-cometkiwi-da)
  - Does not require reference translations
  - Evaluates source-translation quality directly
  - Note: Requires Python 3.11+ and HuggingFace token for gated model access

#### LLM-as-Judge
- **Judge Model**: GPT-4o with pairwise comparison
- **Evaluation Criteria**: Fluency, adequacy, completeness
- **Output Schema**: Winner (text_only | text_image | tie) + confidence (high/medium/low) + reasoning
- **Configuration**: Requires API_KEY and MODEL_NAME in .env file

#### Hardware
- **GPU**: ~8GB VRAM for Qwen2-VL-2B-Instruct (MPS or CUDA)
- **Inference**: ~2-5 seconds per sample

## Usage Scenarios

### Scenario 1: Quick Validation (5-10 minutes)
```bash
python -m pipeline.run --stage all --limit 5
# Validates pipeline on small sample
# Output: artifacts/summaries/evaluation_report.txt
```

### Scenario 2: Full Experiment
```bash
# Step 1: Inference (requires GPU/MPS)
python -m pipeline.run --stage inference --device mps

# Step 2: CometKiwi scoring
python -m pipeline.run --stage scoring

# Step 3: LLM-as-a-Judge (requires .env with API_KEY)
python -m pipeline.run --stage judging

# Step 4: Generate summaries
python -m pipeline.run --stage summary
```

### Scenario 3: Generate Reports Only
```bash
# If inference/scoring/judging already complete:
python -m pipeline.run --stage summary
# Generates: evaluation_report.txt, cometkiwi_summary.json, llm_judge_summary.json
```

### Scenario 4: Environment Setup
```bash
# Create .env file for LLM Judge
echo "API_BASE_URL=https://api.openai.com/v1" > .env
echo "API_KEY=sk-xxx" >> .env
echo "MODEL_NAME=gpt-4o" >> .env
```

## Experiment Output Format

### Evaluation Report (`artifacts/summaries/evaluation_report.txt`)
```
=== Multimodal MT Evaluation Report ===
Evaluation Date: 2026-01-20
Total Samples: 265

--- Overall Statistics ---
Metric: CometKiwi (wmt22-cometkiwi-da)
Text-Only Mean: 0.4628
Text+Image Mean: 0.4795
Delta: +0.0167

--- LLM-as-a-Judge (GPT-4o) ---
Text-Image Wins: 169 (63.8%)
Text-Only Wins: 90 (34.0%)
Ties: 6 (2.3%)

--- Top Language Improvements ---
ro_RO: +0.2178
de_DE: +0.0958
hi_IN: +0.0790

--- Error Taxonomy ---
Text-Only Issues: 30 (11.3%)
  - Language mixing: 24 cases
  - Repetition: 6 cases
  
Text+Image Issues: 14 (5.3%)
  - Language mixing: 14 cases
```

### JSON Summaries
- `cometkiwi_summary.json` - Detailed CometKiwi scores by language
- `llm_judge_summary.json` - LLM judge results with confidence distribution

## Troubleshooting

### Memory Issues
```bash
# Fall back to CPU if GPU memory insufficient:
python -m pipeline.run --stage inference --device cpu
```

### Resume After Interruption
```bash
# Pipeline automatically resumes - just rerun the same command
python -m pipeline.run --stage inference
```

## License

MIT License - See LICENSE file for details.

## References

- **Qwen2-VL**: [Qwen/Qwen2-VL on Hugging Face](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- **CometKiwi**: [Unbabel/COMET](https://github.com/Unbabel/COMET)
- **WMT2025 Shared Task**: [WMT General MT](https://www.statmt.org/wmt25/)
- **Evaluation Methodology**: Kocmi et al. (2024) WMT Metrics Shared Task