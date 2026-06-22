# Phase 2 – Image Preprocessing

## Objective
Prepare Telugu OCR images for recognition by applying image enhancement and cleaning techniques.

## Preprocessing Steps
1. Convert image to grayscale
2. Noise reduction using Non-Local Means Denoising (mild, h=10)
3. Otsu's automatic global thresholding for binarization

::: Fixed bug — see presentation_notes.md :::
The original pipeline included a 4th step (global histogram
equalization) between denoising and thresholding. On this corpus's
typically high-contrast scans, that step over-amplified contrast and
caused adjacent character strokes to bleed together — Tesseract
returned **empty output** on otherwise clean pages as a result (verified:
CER went from 1.00/total-failure to 0.276 on a test page after removing
it). Histogram equalization was removed and adaptive thresholding was
replaced with Otsu's method, which derives the threshold from each
page's actual histogram instead of applying a blind global stretch.

## Inputs
data/ground_truth/*.jpg

(or a larger sample from data/ocr_sample/*.jpg — see phase3/0_sample_corpus_for_ocr.py)

## Outputs
phase2/outputs/preprocessed_images/*.png

phase2/outputs/preprocessing_report.csv

phase2/outputs/sample_comparisons/*.png

## Run

python phase2/run_phase2.py

python phase2/1_preprocess_images.py

python phase2/1_preprocess_images.py --input-dir data/ocr_sample --out-dir phase2/outputs/preprocessed_sample

## Dependencies

pip install opencv-python pandas tqdm
