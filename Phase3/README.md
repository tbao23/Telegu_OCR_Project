# Phase 3 – OCR Evaluation and Model Comparison

## Objective

Evaluate OCR performance on Telugu text by comparing OCR model outputs against manually curated ground truth transcriptions.

## OCR Models

* Tesseract OCR (Telugu language pack)
* EasyOCR (Telugu model)

## Evaluation Metrics

### Character Error Rate (CER)

Measures character-level transcription errors.

CER = (Substitutions + Insertions + Deletions) / Total Characters

Lower CER indicates better OCR performance.

### Word Error Rate (WER)

Measures word-level transcription errors.

WER = (Substitutions + Insertions + Deletions) / Total Words

Lower WER indicates better OCR performance.

## Inputs

Ground Truth:

data/ground_truth/*.txt

Preprocessed Images:

phase2/outputs/preprocessed_images/*.png

## Outputs

OCR Text:

phase3/outputs/tesseract/*.txt

phase3/outputs/easyocr/*.txt

Evaluation Results:

phase3/outputs/metrics/ocr_comparison_results.csv

phase3/outputs/metrics/ocr_summary.csv

## Run OCR

python phase3/5_run_ocr.py --model tesseract

python phase3/5_run_ocr.py --model easyocr

## Compare Models

python phase3/6_compare_ocr_models.py

## Dependencies

pip install pytesseract pillow easyocr pandas tqdm
