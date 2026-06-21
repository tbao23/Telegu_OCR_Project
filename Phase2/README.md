# Phase 2 – Image Preprocessing

## Objective
Prepare Telugu OCR images for recognition by applying image enhancement and cleaning techniques.

## Preprocessing Steps
1. Convert image to grayscale
2. Noise reduction using Non-Local Means Denoising
3. Histogram Equalization for contrast enhancement
4. Adaptive Thresholding for binarization

## Inputs
data/ground_truth/*.jpg

## Outputs
phase2/outputs/preprocessed_images/*.png

phase2/outputs/preprocessing_report.csv

phase2/outputs/sample_comparisons/*.png

## Run

python phase2/4_preprocess_images.py

## Dependencies

pip install opencv-python pandas tqdm
