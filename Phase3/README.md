# Phase 3 – OCR Pipeline and Model Comparison

## Objective

Build a production-grade OCR pipeline using two or more models — at least one
classical baseline and at least one LLM-based engine — and evaluate their
comparative performance against ground truth.

## OCR Models

* **Tesseract OCR** (Telugu language pack) — free baseline
* **Qwen3-VL** via local Ollama — vision-LLM OCR, free, no API key
* EasyOCR (Telugu model) — optional third option

## Evaluation Metrics

### Character Error Rate (CER)

Measures character-level transcription errors.

CER = (Substitutions + Insertions + Deletions) / Total Characters

Lower CER indicates better OCR performance.

### Word Error Rate (WER)

Measures word-level transcription errors.

WER = (Substitutions + Insertions + Deletions) / Total Words

Lower WER indicates better OCR performance.

Text is Unicode-NFC normalized before comparison, so differing Unicode
representations of the same visual character aren't miscounted as errors.

## Pipeline Steps

### Or run everything at once:

```
python phase3/run_phase3.py
python phase3/run_phase3.py --use-corpus-sample --sample-size 100
python phase3/run_phase3.py --models tesseract
```

Chains sampling (if requested) → preprocessing (pulled in from Phase 2) →
OCR for each requested model → comparison, all in one command. Resumable
(skips steps whose output already exists; `--force` to redo) and
hard-fails immediately with a clear message if Ollama isn't ready for
`qwen3vl`.

### Or run each step individually:

### 0. Sample pages from the full corpus (for scaling beyond the 40-page Phase 1 sample)

The Phase 1 ground truth (40 pages) is for manual annotation / CER baseline.
Phase 4 needs 100+ pages and the final deliverable needs 500+ — this script
draws a larger, independent, quality-stratified sample directly from the full
corpus, reusing existing Wikisource transcriptions as reference text.

```
python phase3/0_sample_corpus_for_ocr.py --corpus-dir data/corpus --n 100 --out data/ocr_sample
```

### 1. Run OCR

```
python phase3/1_run_ocr.py --input-dir phase2/outputs/preprocessed_images --model tesseract
python phase3/1_run_ocr.py --input-dir phase2/outputs/preprocessed_images --model qwen3vl
python phase3/1_run_ocr.py --input-dir phase2/outputs/preprocessed_images --model easyocr
```

`qwen3vl` requires Ollama running locally with the vision model pulled:
```
ollama pull qwen3-vl:8b
```

`tesseract` requires Tesseract installed and on PATH, or pass
`--tesseract-path "C:\Program Files\Tesseract-OCR\tesseract.exe"` (or set the
`TESSERACT_PATH` environment variable).

### 2. Compare models against ground truth

```
python phase3/2_compare_ocr_models.py --reference-dir data/ground_truth --ocr-root phase3/outputs
```

## Inputs

Ground Truth / Reference text:

data/ground_truth/*.txt (40-page Phase 1 sample)

data/ocr_sample/*.txt (larger sample, see Step 0 above)

Preprocessed Images:

phase2/outputs/preprocessed_images/*.png

## Outputs

OCR Text:

phase3/outputs/tesseract/*.txt

phase3/outputs/qwen3vl/*.txt

phase3/outputs/easyocr/*.txt

Evaluation Results:

phase3/outputs/metrics/ocr_comparison_results.csv

phase3/outputs/metrics/ocr_summary.csv

## Dependencies

pip install pytesseract pillow easyocr pandas tqdm requests
