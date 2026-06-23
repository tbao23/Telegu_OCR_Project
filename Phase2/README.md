# Phase 2 – Image Preprocessing

## Objective
Prepare Telugu OCR images for recognition by applying image enhancement and cleaning techniques.

## Preprocessing Steps
1. Convert image to grayscale
2. Noise reduction using Non-Local Means Denoising (mild, h=10)
3. Otsu's automatic global thresholding for binarization

## Two Real Bugs Found and Fixed Here

::: Bug 1 — Unicode filenames silently dropped pages :::
`cv2.imread()`/`cv2.imwrite()` can't handle non-ASCII file paths on
Windows — a documented OpenCV limitation. This silently failed on 22.5%
of pages (9 of 40), and every single failure had a Telugu-script
filename while every success had a Latin-script one — specifically and
systematically dropping Telugu-named content from a Telugu OCR
pipeline. Fixed by decoding image bytes manually (`np.fromfile` +
`cv2.imdecode`/`cv2.imencode`) instead of letting OpenCV touch the file
path directly. Result: 31/40 → 40/40 pages processed successfully.

::: Bug 2 — histogram equalization was destroying clean scans :::
The original pipeline included a 4th step (global histogram
equalization) between denoising and thresholding. On this corpus's
typically high-contrast scans, that step over-amplified contrast and
caused adjacent character strokes to bleed together — Tesseract
returned **empty output** on otherwise clean pages as a result (verified:
CER went from 1.00/total-failure to 0.276 on a test page after removing
it). Histogram equalization was removed and adaptive thresholding was
replaced with Otsu's method, which derives the threshold from each
page's actual histogram instead of applying a blind global stretch.

See `presentation_notes.md` (project root) for the full writeup of both,
including the diagnostic methodology used to isolate each bug.

## Preprocessing Impact — Quantifying the Effect (Dimensions 2 & 5)

A single anecdotal test page isn't sufficient evidence that preprocessing
helps — the assignment requires a real, measured before/after comparison.
These two scripts run real OCR on both the raw and preprocessed versions
of the SAME 40 ground-truth pages, with both models, producing a
genuine measured CER delta:

```
python phase2/2_convert_raw_baseline.py --input-dir data/ground_truth --out-dir outputs/phase2/raw_baseline
python phase3/1_run_ocr.py --input-dir outputs/phase2/raw_baseline --model claude --out-root outputs/phase2/raw_baseline_ocr
python phase3/1_run_ocr.py --input-dir outputs/phase2/preprocessed_images --model claude --out-root outputs/phase2/preprocessed_ocr
python phase3/2_compare_ocr_models.py --reference-dir data/ground_truth --ocr-root outputs/phase2/raw_baseline_ocr --metrics-dir outputs/phase2/raw_baseline_ocr/metrics
python phase3/2_compare_ocr_models.py --reference-dir data/ground_truth --ocr-root outputs/phase2/preprocessed_ocr --metrics-dir outputs/phase2/preprocessed_ocr/metrics
python phase2/3_preprocessing_impact.py --raw-summary outputs/phase2/raw_baseline_ocr/metrics/ocr_summary.csv --preprocessed-summary outputs/phase2/preprocessed_ocr/metrics/ocr_summary.csv
```

(`run_all.py` runs all of this automatically as part of the full pipeline
— see root README. The manual sequence above is for running just this
piece on its own.)

This project's actual real result: preprocessing had no measurable
benefit for Tesseract (0.6% improvement, noise-level) and slightly
*worsened* Claude's accuracy (10.2% relative) — once the destructive
equalizeHist step was already removed. See `final_report.qmd` for the
full discussion.

## Inputs
data/ground_truth/*.jpg

(or a larger sample from data/ocr_sample/*.jpg — see phase3/0_sample_corpus_for_ocr.py)

## Outputs
outputs/phase2/preprocessed_images/*.png

outputs/phase2/preprocessing_report.csv

outputs/phase2/sample_comparisons/*.png (before/after images — at least 10, per assignment requirement)

outputs/phase2/raw_baseline/*.png, outputs/phase2/raw_baseline_ocr/, outputs/phase2/preprocessed_ocr/ (preprocessing impact comparison)

outputs/phase2/preprocessing_impact.csv, outputs/phase2/preprocessing_impact.json

## Run

python phase2/run_phase2.py

python phase2/1_preprocess_images.py

python phase2/1_preprocess_images.py --input-dir data/ocr_sample --out-dir outputs/phase2/preprocessed_sample

## Dependencies

pip install opencv-python pandas tqdm
