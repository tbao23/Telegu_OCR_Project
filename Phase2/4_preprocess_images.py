"""
Phase 2 — Image Preprocessing Pipeline

Input:
  data/ground_truth/*.jpg

Output:
  phase2/outputs/preprocessed_images/*.png
  phase2/outputs/preprocessing_report.csv
  phase2/outputs/sample_comparisons/
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def preprocess_image(image_path: Path, out_path: Path) -> dict:
    image = cv2.imread(str(image_path))

    if image is None:
        return {
            "filename": image_path.name,
            "status": "failed",
            "notes": "Could not read image"
        }

    original_height, original_width = image.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

    # Improve contrast
    equalized = cv2.equalizeHist(denoised)

    # Binarize image
    binary = cv2.adaptiveThreshold(
        equalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15
    )

    # Save processed image
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), binary)

    return {
        "filename": image_path.name,
        "output_file": out_path.name,
        "status": "processed",
        "original_width": original_width,
        "original_height": original_height,
        "processing_steps": "grayscale, denoise, contrast_equalization, adaptive_threshold"
    }


def save_comparison(original_path: Path, processed_path: Path, comparison_path: Path) -> None:
    original = cv2.imread(str(original_path))
    processed = cv2.imread(str(processed_path))

    if original is None or processed is None:
        return

    if len(processed.shape) == 2:
        processed_color = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    else:
        processed_color = processed.copy()

    height = min(original.shape[0], processed_color.shape[0])
    original_resized = cv2.resize(
        original,
        (int(original.shape[1] * height / original.shape[0]), height)
    )
    processed_resized = cv2.resize(
        processed_color,
        (int(processed_color.shape[1] * height / processed_color.shape[0]), height)
    )

    comparison = np.hstack([original_resized, processed_resized])
    cv2.imwrite(str(comparison_path), comparison)


def main():
    parser = argparse.ArgumentParser(description="Phase 2 — Preprocess Telugu OCR images")
    parser.add_argument("--input-dir", type=Path, default=Path("data/ground_truth"))
    parser.add_argument("--out-dir", type=Path, default=Path("phase2/outputs/preprocessed_images"))
    parser.add_argument("--report", type=Path, default=Path("phase2/outputs/preprocessing_report.csv"))
    parser.add_argument("--comparison-dir", type=Path, default=Path("phase2/outputs/sample_comparisons"))
    parser.add_argument("--num-comparisons", type=int, default=5)
    args = parser.parse_args()

    image_files = sorted(args.input_dir.glob("*.jpg"))

    if not image_files:
        raise SystemExit(f"No .jpg images found in {args.input_dir}")

    records = []

    for image_path in tqdm(image_files, desc="Preprocessing images"):
        out_path = args.out_dir / f"{image_path.stem}.png"
        record = preprocess_image(image_path, out_path)
        records.append(record)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.report, index=False, encoding="utf-8-sig")

    for image_path in image_files[:args.num_comparisons]:
        processed_path = args.out_dir / f"{image_path.stem}.png"
        comparison_path = args.comparison_dir / f"{image_path.stem}_comparison.png"
        save_comparison(image_path, processed_path, comparison_path)

    print("\nPhase 2 complete.")
    print(f"Processed images saved to: {args.out_dir}")
    print(f"Report saved to: {args.report}")
    print(f"Sample comparisons saved to: {args.comparison_dir}")


if __name__ == "__main__":
    main()
