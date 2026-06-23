# Phase 3 – OCR Pipeline and Model Comparison

## Objective

Build a production-grade OCR pipeline using two or more models — at least one
classical baseline and at least one LLM-based engine — and evaluate their
comparative performance against ground truth.

## OCR Models

* **Claude** (Anthropic, vision API) — this project's primary LLM-based OCR
  model. See the root README and `final_report.qmd`'s Phase 3 Model
  Selection discussion for why Claude was chosen over Gemini/Qwen3-VL —
  short version: Gemini's free tier hit content-safety blocking on
  copyrighted-era Telugu literature and a platform authentication bug
  during the project's final week; Claude doesn't have those issues.
* **Tesseract OCR** (Telugu language pack) — free, local classical baseline
* **Qwen3-VL** via local Ollama — free vision-LLM alternative, no API key,
  but noticeably slower on CPU-only hardware (real measured: this project
  found Ollama vision inference impractical at the needed scale)
* **Gemini** (Google AI Studio) — still supported in code, not used for this
  project's actual reported results (see caveats above)
* EasyOCR (Telugu model) — optional additional classical baseline

## Evaluation Metrics

### Character Error Rate (CER)

Measures character-level transcription errors.

CER = (Substitutions + Insertions + Deletions) / Total Characters

Lower CER indicates better OCR performance.

### Word Error Rate (WER)

Measures word-level transcription errors.

WER = (Substitutions + Insertions + Deletions) / Total Words

Lower WER indicates better OCR performance.

Computed via the `jiwer` library (not hand-rolled — see AI Tool Usage
Disclosure in `final_report.qmd` for why this matters). Text is
Unicode-NFC normalized before comparison, so differing Unicode
representations of the same visual character aren't miscounted as errors.
Pages with very short reference text (<50 chars by default) are flagged
separately, since CER is statistically unstable on them — summaries are
reported both including and excluding flagged pages.

## Pipeline Steps

### Recommended: run everything via the project-root orchestrator

```
python run_all.py --keep-ground-truth --sample-size 500 --validation-limit 100 --models claude,tesseract --force
```

This runs Phase 3 as part of the full 5-phase pipeline, including the
raw-vs-preprocessed comparison (see Phase 2's README) before it. See the
root `README.md` Quickstart for the full explanation of every flag.

### Or run just Phase 3 on its own:

```
python phase3/run_phase3.py --use-corpus-sample --sample-size 500 --models claude,tesseract
python phase3/run_phase3.py --models tesseract               # tesseract only, on the 40-page ground truth
```

Chains sampling (if requested) → preprocessing (pulled in from Phase 2) →
OCR for each requested model → comparison, all in one command. Resumable
(skips steps whose output already exists, comparing actual file *counts*
not just existence — re-running with a smaller `--sample-size` than a
previous run correctly clears stale leftovers rather than silently
processing extra pages) and hard-fails immediately with a clear message if
a required backend isn't ready (e.g. Ollama for `qwen3vl`, no Anthropic key
set for `claude`).

### Or run each step individually:

#### 0. Sample pages from the full corpus (optional — for scaling beyond the 40-page Phase 1 sample)

The Phase 1 ground truth (40 pages) is for manual annotation / CER baseline.
Phase 4 needs 100+ pages and the final deliverable needs 500+ — this script
draws a larger, independent, quality-stratified sample directly from the full
corpus, reusing existing Wikisource transcriptions as reference text. It
clears any stale leftover files in `--out` before writing a new sample, so
re-running with a different `--n` never leaves old pages mixed in.

```
python phase3/0_sample_corpus_for_ocr.py --corpus-dir data/corpus --n 500 --out data/ocr_sample
```

#### 1. Run OCR

```
python phase3/1_run_ocr.py --input-dir outputs/phase2/preprocessed_sample --model claude --out-root outputs/phase3
python phase3/1_run_ocr.py --input-dir outputs/phase2/preprocessed_sample --model tesseract --out-root outputs/phase3
python phase3/1_run_ocr.py --input-dir outputs/phase2/preprocessed_sample --model qwen3vl --out-root outputs/phase3
python phase3/1_run_ocr.py --input-dir outputs/phase2/preprocessed_sample --model easyocr --out-root outputs/phase3
```

Resumable **per page**: if a page's output `.txt` already exists, it's
reused rather than re-run — critical for the paid `claude` backend, since
an interrupted run never forces a costly full re-pay on restart. Pass
`--force` to redo every page regardless.

`claude` requires `ANTHROPIC_API_KEY` set (see root README Prerequisites).
`qwen3vl` requires Ollama running locally with the vision model pulled:
```
ollama pull qwen3-vl:8b
```
`tesseract` requires Tesseract installed and on PATH, or pass
`--tesseract-path "C:\Program Files\Tesseract-OCR\tesseract.exe"` (or set the
`TESSERACT_PATH` environment variable).

#### 2. Compare models against ground truth

```
python phase3/2_compare_ocr_models.py --reference-dir data/ocr_sample --ocr-root outputs/phase3 --metrics-dir outputs/phase3/metrics
```

## Preprocessing Impact (raw vs. preprocessed OCR — see Phase 2)

`phase2/2_convert_raw_baseline.py` and `phase2/3_preprocessing_impact.py`
run real OCR on both the raw and preprocessed versions of the SAME 40
ground-truth pages, producing the quantified "does preprocessing actually
help" finding the assignment requires (Dimensions 2 and 5). These live in
`phase2/` since they're specifically about measuring preprocessing's
contribution, but reuse this phase's `1_run_ocr.py` and
`2_compare_ocr_models.py` scripts internally — see `phase2/README.md`.

## Inputs

Ground Truth / Reference text:

data/ground_truth/*.txt (40-page Phase 1 sample)

data/ocr_sample/*.txt (larger corpus-wide sample, see Step 0 above)

Preprocessed Images:

outputs/phase2/preprocessed_images/*.png (40-page sample)

outputs/phase2/preprocessed_sample/*.png (larger corpus-wide sample)

## Outputs

OCR Text:

outputs/phase3/claude/*.txt

outputs/phase3/tesseract/*.txt

outputs/phase3/qwen3vl/*.txt (if used)

outputs/phase3/easyocr/*.txt (if used)

Evaluation Results:

outputs/phase3/metrics/ocr_comparison_results.csv

outputs/phase3/metrics/ocr_summary.csv

## Dependencies

pip install pytesseract pillow easyocr pandas tqdm requests anthropic jiwer
