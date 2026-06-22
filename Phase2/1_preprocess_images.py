"""
phase2/1_preprocess_images.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 2 — Image Preprocessing Pipeline

Input:
  data/ground_truth/*.jpg

Output:
  outputs/phase2/preprocessed_images/*.png
  outputs/phase2/preprocessing_report.csv
  outputs/phase2/sample_comparisons/
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def imread_unicode(path: Path):
    """
    cv2.imread() silently fails (returns None) on Windows when the path
    contains non-ASCII characters — a real problem for a Telugu OCR
    project, since book titles are often in Telugu script. This reads
    the raw bytes first (which handles Unicode paths fine) then decodes
    with OpenCV, sidestepping the bug entirely.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    """Unicode-safe counterpart to imread_unicode — same underlying issue
    affects cv2.imwrite() on Windows for non-ASCII paths."""
    ext = Path(path).suffix or ".png"
    success, encoded = cv2.imencode(ext, image)
    if not success:
        return False
    encoded.tofile(str(path))
    return True


def preprocess_image(image_path: Path, out_path: Path) -> dict:
    image = imread_unicode(image_path)

    if image is None:
        return {
            "filename": image_path.name,
            "status": "failed",
            "notes": "Could not read image"
        }

    original_height, original_width = image.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Denoise (h=10 is the literature-standard strength; h=30 was 3x too
    # aggressive and blurred fine Telugu strokes/diacritics before they
    # ever reached thresholding)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # FIX: cv2.equalizeHist() applies GLOBAL histogram equalization, which
    # over-amplifies contrast on already-clean scans (this corpus's pages
    # are typically high-contrast: mean brightness ~239, std ~57). On a
    # page that already has good separation between text and background,
    # this stretches mid-tones so aggressively that adjacent character
    # strokes bleed together into unrecognizable blobs — verified this
    # was THE cause of Tesseract returning empty output on otherwise
    # clean source pages (CER dropped from 1.00/total-failure to 0.276
    # on a test page after removing this step). Skipping it entirely and
    # using Otsu's automatic global threshold instead, which adapts to
    # each page's actual histogram rather than blindly stretching it.
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Save processed image
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite_unicode(out_path, binary)

    return {
        "filename": image_path.name,
        "output_file": out_path.name,
        "status": "processed",
        "original_width": original_width,
        "original_height": original_height,
        "processing_steps": "grayscale, denoise, otsu_threshold"
    }


def save_comparison(original_path: Path, processed_path: Path, comparison_path: Path) -> None:
    original = imread_unicode(original_path)
    processed = imread_unicode(processed_path)

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
    comparison_path.parent.mkdir(parents=True, exist_ok=True)  # FIX: was missing — cv2.imwrite
                                                                 # fails silently if the folder
                                                                 # doesn't exist, which is why
                                                                 # sample_comparisons/ never appeared
    imwrite_unicode(comparison_path, comparison)


def main():
    parser = argparse.ArgumentParser(description="Phase 2 — Preprocess Telugu OCR images")
    parser.add_argument("--input-dir", type=Path, default=Path("data/ground_truth"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/phase2/preprocessed_images"))
    parser.add_argument("--report", type=Path, default=Path("outputs/phase2/preprocessing_report.csv"))
    parser.add_argument("--comparison-dir", type=Path, default=Path("outputs/phase2/sample_comparisons"))
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

    successful = [r["filename"] for r in records if r.get("status") == "processed"]
    n_failed = len(records) - len(successful)
    if n_failed > 0:
        print(f"\n[NOTICE] {n_failed} image(s) failed preprocessing (unreadable file) — "
              "see the report CSV's 'notes' column. Skipped for comparison generation.")

    for filename in successful[:args.num_comparisons]:
        image_path = args.input_dir / filename
        processed_path = args.out_dir / f"{Path(filename).stem}.png"
        comparison_path = args.comparison_dir / f"{Path(filename).stem}_comparison.png"
        save_comparison(image_path, processed_path, comparison_path)

    print("\nPhase 2 complete.")
    print(f"Processed images saved to: {args.out_dir}")
    print(f"Report saved to: {args.report}")
    print(f"Sample comparisons saved to: {args.comparison_dir}")
    print("\nNext step:")
    print(f"  python phase3/1_run_ocr.py --input-dir {args.out_dir} --model tesseract")


if __name__ == "__main__":
    main()
