# Phase 4 – LLM-Assisted Validation Framework

## Objective

Combine classical evaluation metrics (CER/WER) with LLM-based linguistic
validation that requires no ground truth — and verify, through calibration
analysis, whether the LLM scores actually track real OCR accuracy.

## Validation Methods

* **Classical metrics** — CER/WER via `jiwer`, against the Phase 1 ground truth
* **Method A — Fluency scoring**: LLM rates OCR output 1–5 for linguistic quality
* **Method B — Error detection**: LLM flags specific implausible words/sequences
* **Method C — Cross-model agreement**: compares two models' output on the same pages
* **Calibration**: Pearson/Spearman correlation between LLM scores and real CER

LLM backend is configurable (`--backend ollama|anthropic|openai|gemini`) —
see `phase4/llm_backends.py`. Ollama (local, free) is the default.

## Pipeline Steps

### 0. (Test-only) Generate synthetic OCR output

Until real Phase 3 OCR output exists, this stands in so the rest of the
pipeline can be built/verified today:

```
python phase4/0_make_synthetic_test_data.py --ground-truth data/ground_truth --out data/synthetic_ocr_output/model_a --error-rate 0.08
```

### 1. Classical metrics (CER/WER)

```
python phase4/1_compute_cer_wer.py --ground-truth data/ground_truth --ocr-output data/synthetic_ocr_output/model_a --model-name model_a
```

### 2. LLM fluency scoring (Method A)

```
python phase4/2_llm_fluency_score.py --ocr-output data/synthetic_ocr_output/model_a --model-name model_a
```

### 3. LLM error detection (Method B)

```
python phase4/3_llm_error_detection.py --ocr-output data/synthetic_ocr_output/model_a --model-name model_a
```

### 4. Cross-model agreement (Method C)

```
python phase4/4_cross_model_agreement.py --output-a data/synthetic_ocr_output/model_a --output-b data/synthetic_ocr_output/model_b
```

### 5. Calibration analysis

```
python phase4/5_calibration_analysis.py --cer-wer-csv outputs/phase4/cer_wer_model_a.csv --fluency-csv outputs/phase4/fluency_model_a.csv
```

### Or run all six steps at once:

```
python phase4/run_phase4.py
```

Resumable — skips any step whose output already exists (use `--force` to
redo). Hard-fails immediately with a clear message if the chosen LLM
backend isn't ready (Ollama not running / API key missing), rather than
silently skipping.

## Known Limitations

* Local Ollama models (Qwen3) have noticeably weaker Telugu fluency than
  Claude/GPT-4o — a documented cost-vs-quality tradeoff, not a bug.
* Qwen3/DeepSeek-style "thinking" models share their output token budget
  between reasoning and the final answer — `llm_backends.py` passes
  `"think": false` to avoid truncated/empty responses on longer prompts.

## Inputs

```
data/ground_truth/*.txt            Phase 1 ground truth (reference text)
data/synthetic_ocr_output/         TEST-ONLY stand-in for Phase 3 output
```

Once real Phase 3 OCR output exists, point `--ocr-output` at it instead —
same filenames, same interface, zero code changes needed.

## Outputs

```
outputs/phase4/cer_wer_<model>.csv
outputs/phase4/cer_wer_<model>_summary.json
outputs/phase4/fluency_<model>.csv
outputs/phase4/error_detection_<model>.csv
outputs/phase4/cross_model_agreement.csv
outputs/phase4/calibration_summary.json
outputs/phase4/calibration_scatter.png
```

## Dependencies

pip install jiwer scipy pandas matplotlib requests

Plus whichever LLM backend SDK you use: `anthropic`, `openai`, or
`google-generativeai` (Ollama needs none of these — see root README
Prerequisites table for install link).
