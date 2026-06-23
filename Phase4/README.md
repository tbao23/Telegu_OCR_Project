# Phase 4 – LLM-Assisted Validation Framework

## Objective

Combine classical evaluation metrics (CER/WER) with LLM-based linguistic
validation that requires no ground truth — and verify, through calibration
analysis, whether the LLM scores actually track real OCR accuracy.

## Validation Methods

* **Classical metrics** — CER/WER via `jiwer`, against real Phase 3 OCR output
* **Method A — Fluency scoring**: LLM rates OCR output 1–5 for linguistic quality
* **Method B — Error detection**: LLM flags specific implausible words/sequences
* **Method C — Cross-model agreement**: compares two models' output on the same pages
* **Calibration**: Pearson/Spearman correlation between LLM scores and real CER

LLM backend is configurable (`--backend ollama|anthropic|openai|gemini`) —
see `llm_backends.py` (project root). **Default is `anthropic`**, using
Claude Haiku specifically (cheaper/faster than Sonnet, appropriate for this
lighter judging task — see `llm_backends.py`'s `DEFAULT_MODELS`). Ollama
was the original default but was found too slow on CPU-only hardware at the
100+ page scale this phase requires; pass `--backend ollama` to go back to
fully local/free if that tradeoff works for your hardware.

## Pipeline Steps

### Recommended: run via the project-root orchestrator (does all 5 phases)

```
python run_all.py --keep-ground-truth --sample-size 500 --validation-limit 100 --models claude,tesseract --force
```

`--validation-limit` caps Methods A/B to 100 pages (the assignment's stated
minimum) even though `--sample-size` (Phase 3) might be 500 — there's no
rubric benefit to spending API quota validating all 500 when 100 satisfies
the requirement. See root `README.md` for the full flag explanation.

### Or run just Phase 4 on real Phase 3 output:

```
python phase4/run_phase4.py --ground-truth data/ocr_sample --ocr-output-a outputs/phase3/claude --model-name-a claude --ocr-output-b outputs/phase3/tesseract --model-name-b tesseract --skip-synthetic --validation-limit 100
```

Resumable — skips any step whose output already exists (use `--force` to
redo). Hard-fails immediately with a clear message if the chosen LLM
backend isn't ready (Ollama not running / API key missing), rather than
silently skipping. CER/WER results computed before a failure point are
still saved.

### Or run each step individually:

#### 0. (Test-only) Generate synthetic OCR output

Useful for testing the pipeline before real Phase 3 output exists, or for
quick dev iteration without spending API calls:

```
python phase4/0_make_synthetic_test_data.py --ground-truth data/ground_truth --out data/synthetic_ocr_output/model_a --error-rate 0.08
```

#### 1. Classical metrics (CER/WER)

```
python phase4/1_compute_cer_wer.py --ground-truth data/ocr_sample --ocr-output outputs/phase3/claude --model-name claude
```

Flags pages with reference text under 50 characters as statistically
unstable for CER (a tiny denominator can produce an extreme ratio from
modest OCR output) — summary is reported both including and excluding
these pages.

#### 2. LLM fluency scoring (Method A)

```
python phase4/2_llm_fluency_score.py --ocr-output outputs/phase3/claude --model-name claude --backend anthropic --limit 100
```

#### 3. LLM error detection (Method B)

```
python phase4/3_llm_error_detection.py --ocr-output outputs/phase3/claude --model-name claude --backend anthropic --limit 100
```

#### 4. Cross-model agreement (Method C)

```
python phase4/4_cross_model_agreement.py --output-a outputs/phase3/claude --output-b outputs/phase3/tesseract
```

#### 5. Calibration analysis

```
python phase4/5_calibration_analysis.py --cer-wer-csv outputs/phase4/cer_wer_claude.csv --fluency-csv outputs/phase4/fluency_claude.csv
```

Also excludes short-reference-flagged pages from the correlation
calculation, for the same reason as Step 1 above.

## Multi-Key Setup (optional, recommended for paid backends)

Both Gemini and Anthropic backends support automatic key rotation and
retry/backoff — see `llm_backends.py`'s module docstring for the full
explanation and environment variable names (`ANTHROPIC_API_KEY_BACKUP1`,
`GOOGLE_API_KEY_PHASE4`, etc.). Not required — a single `ANTHROPIC_API_KEY`
works fine — but useful if you hit rate limits at scale.

## Known Limitations

* Local Ollama models (Qwen3) have noticeably weaker Telugu fluency than
  Claude — a documented cost-vs-quality tradeoff, not a bug.
* Qwen3/DeepSeek-style "thinking" models share their output token budget
  between reasoning and the final answer — `llm_backends.py` passes
  `"think": false` to avoid truncated/empty responses on longer prompts.
* This project's actual calibration results showed a real but moderate
  correlation between LLM fluency score and true CER (Pearson r ≈ -0.33,
  Spearman r ≈ -0.60) — see `final_report.qmd` for the full discussion of
  why fluency and character-level accuracy capture related but distinct
  signals.

## Inputs

```
data/ocr_sample/*.txt                      Reference text (large corpus sample)
data/ground_truth/*.txt                    Reference text (40-page Phase 1 sample)
outputs/phase3/<model>/*.txt               Real OCR output from Phase 3
data/synthetic_ocr_output/                 TEST-ONLY stand-in, not used for real results
```

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

pip install jiwer scipy pandas matplotlib requests anthropic

Plus whichever other LLM backend SDK you use: `openai`, or `google-genai`
(**not** `google-generativeai` — that package is deprecated; Ollama needs
none of these — see root README Prerequisites table for install links).
