# 4-Day Multimodal MT Experiment Guide

## 🎯 Experiment Goal

Compare Qwen2-VL-2B translation quality with and without image context for the WMT2025 multimodal MT task.

**Key Principles:**
- ✅ Inference only (no training)
- ✅ Single model (Qwen2-VL-2B)
- ✅ Fixed sample size (50 or 100)
- ✅ All results saved incrementally

---

## 📋 Prerequisites

### 1. Install Dependencies

```bash
cd /Users/yue/Documents/code/nlpproject/multimodal-mt-eval
pip install -e .
pip install -r requirements.txt
```

### 2. Set Up API Keys (for Day 3 Judge)

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### 3. Configure Experiment

Edit `config/experiment.yaml`:
- Set `data.num_samples` to 50 or 100
- Set `model.device` to "cuda" or "cpu"
- **⚠️ After Day 2 starts, DO NOT modify this file!**

---

## 📅 Day-by-Day Execution

### Day 1: Debug with Small Samples (4-5 hours)

#### 1.1 Prepare Data (1-2h)

```bash
python scripts/prepare_data.py
```

**What it does:**
- Downloads/prepares WMT2025 data
- Creates debug subset (5 samples)
- Creates full experiment dataset

**Output:**
- `data/debug_samples.jsonl` (5 samples for testing)
- `data/experiment_data.jsonl` (50/100 samples, frozen)

---

#### 1.2 Text-only Inference (1-2h)

```bash
python scripts/infer_text_only.py
```

**What it does:**
- Loads Qwen2-VL model
- Translates debug samples (text only)
- Saves results incrementally

**Output:**
- `outputs/text_only_debug.jsonl`

**Sanity Check:**
```bash
head -3 outputs/text_only_debug.jsonl
```

---

#### 1.3 Multimodal Inference (1-2h)

```bash
python scripts/infer_multimodal.py
```

**What it does:**
- Translates debug samples (text + image)
- Saves results incrementally

**Output:**
- `outputs/text_image_debug.jsonl`

---

#### 1.4 Manual Comparison (30min)

Open the two output files and compare:
- Are translations different?
- Does image context change output?
- Any obvious errors?

**✅ Day 1 Success Criteria:**
> You can confidently say: "My pipeline works. Scaling up is just a matter of time."

---

### Day 2: Full-Scale Inference (4-8 hours)

⚠️ **BEFORE YOU START:**
1. Verify `config/experiment.yaml` is correct
2. After this point, **NO MORE CONFIG CHANGES**
3. Make sure you have enough disk space and time

---

#### 2.1 Run Full Inference

```bash
python scripts/run_full_inference.py
```

This will:
1. Load full experiment data (50/100 samples)
2. Run text-only inference → `outputs/text_only.jsonl`
3. Run multimodal inference → `outputs/text_image.jsonl`

**Time estimate:**
- 50 samples: 2-4 hours
- 100 samples: 4-8 hours

**⚠️ Important:**
- Results are saved incrementally (防止crash)
- If interrupted, you can resume (check last saved ID)

**✅ Day 2 Success Criteria:**
> You have complete results in `outputs/`. Even if your computer crashes, you can continue analysis.

---

### Day 3: Automatic Evaluation (4-6 hours)

#### 3.1 CometKiwi Evaluation (2-3h)

```bash
python scripts/eval_comet.py
```

**What it does:**
- Loads both translations
- Evaluates with CometKiwi
- Computes delta scores
- Shows top improvements/degradations

**Output:**
- `analysis/comet_scores.csv`

**Expected output:**
```
COMET EVALUATION SUMMARY
Total Samples: 50
Text-only Mean: 0.7234
Multimodal Mean: 0.7456
Delta Mean: +0.0222
Win Rate: 64.0%
```

---

#### 3.2 LLM-as-a-Judge (2-3h)

```bash
python scripts/judge.py
```

**What it does:**
- Uses GPT-4 to judge each pair
- Randomizes order (avoid position bias)
- Saves judgments with reasons

**Output:**
- `analysis/judge_results.csv`

**Note:** Requires OpenAI API key. Can be expensive for 100 samples (~$5-10).

**Alternative:** Manually judge a random subset of 20-30 samples.

---

### Day 4: Error Analysis (4-6 hours)

#### 4.1 Automated Analysis

```bash
python scripts/analyze_errors.py
```

**What it does:**
- Categorizes samples (improvement/degradation)
- Extracts top/bottom cases
- Generates summary report
- Creates annotation template

**Outputs:**
- `analysis/error_cases.csv` - Top improvements and degradations
- `analysis/error_cases_annotation.csv` - Template for manual annotation
- `analysis/summary.txt` - Complete experiment summary

---

#### 4.2 Manual Error Annotation (2-3h)

Open `analysis/error_cases_annotation.csv` and fill in:

**Error Type Categories:**
- `visual_disambiguation` - Image helped disambiguate meaning
- `image_misleading` - Image led to incorrect translation
- `image_irrelevant` - Image didn't affect translation
- `over_interpretation` - Model over-interpreted visual details
- `correct_with_image` - Better translation with visual context
- `other` - Other error types

**Add notes** for each case explaining your judgment.

---

#### 4.3 Final Report

Review `analysis/summary.txt` for complete findings.

**Key Questions to Answer:**
1. When does multimodal input help? (visual disambiguation?)
2. When does it hurt? (image misleading?)
3. What's the overall win rate? (Comet vs Judge agreement?)
4. Is it worth the computational cost?

---

## 📊 Expected Directory Structure After Day 4

```
multimodal-mt-eval/
├── config/
│   └── experiment.yaml         # FROZEN after Day 2
├── data/
│   ├── wmt2025_raw/           # Raw downloaded data
│   ├── debug_samples.jsonl    # Day 1 debug samples
│   └── experiment_data.jsonl  # Full dataset (50/100)
├── outputs/
│   ├── text_only_debug.jsonl  # Day 1 text-only
│   ├── text_image_debug.jsonl # Day 1 multimodal
│   ├── text_only.jsonl        # Day 2 text-only (full)
│   └── text_image.jsonl       # Day 2 multimodal (full)
├── analysis/
│   ├── comet_scores.csv       # Day 3 Comet evaluation
│   ├── judge_results.csv      # Day 3 LLM judge
│   ├── error_cases.csv        # Day 4 top cases
│   ├── error_cases_annotation.csv  # Day 4 manual annotations
│   └── summary.txt            # Day 4 final report
└── scripts/
    ├── prepare_data.py
    ├── infer_text_only.py
    ├── infer_multimodal.py
    ├── run_full_inference.py
    ├── eval_comet.py
    ├── judge.py
    └── analyze_errors.py
```

---

## 🚨 Troubleshooting

### Model Loading Issues

```bash
# Login to Hugging Face
huggingface-cli login

# Check if model is accessible
python -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')"
```

### Out of Memory

- Reduce `batch_size` in `config/experiment.yaml`
- Use smaller sample size (50 instead of 100)
- Run on CPU (slower but more stable)

### API Rate Limits (Day 3 Judge)

- Add `time.sleep()` between calls (already included)
- Use smaller sample size
- Manually judge a subset instead

---

## 📝 Tips for Success

1. **Day 1 is the most important** - Spend time here to debug everything
2. **Save config early** - Once Day 2 starts, no changes allowed
3. **Monitor inference** - Check a few outputs during Day 2 to catch errors early
4. **Incremental saves** - Scripts save after each sample, so crashes are recoverable
5. **Keep notes** - Document any issues or observations in a separate notes.md file

---

## 🎓 What You'll Learn

- ✅ Multimodal LLM inference pipelines
- ✅ Automatic MT evaluation (Comet, LLM judge)
- ✅ Error analysis methodology
- ✅ Experiment design and reproducibility

---

## 📚 Additional Resources

- WMT2025: https://www2.statmt.org/wmt25/
- Qwen2-VL: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
- CometKiwi: https://github.com/Unbabel/COMET

---

**Good luck! 🚀**
