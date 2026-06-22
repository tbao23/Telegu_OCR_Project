"""
phase2/2_convert_raw_baseline.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 2, Step 2 — Raw Baseline Conversion (NO Processing)

Converts ground-truth .jpg scans to .png with ZERO image processing
applied (no grayscale, no denoising, no binarization) — this is
intentionally the "raw" condition for a fair, controlled comparison
against the preprocessed pipeline output.

Why this exists: the assignment requires "quantitative before/after
comparisons demonstrate measurable improvement in OCR accuracy"
(Dimension 2) and "preprocessing contribution is quantified"
(Dimension 5). A single anecdotal example page is not sufficient to
satisfy either — this script, paired with 3_preprocessing_impact.py,
produces a real measured CER delta across the full ground-truth sample.

Uses the same Unicode-safe image I/O as 1_preprocess_images.py (plain
cv2.imread/imwrite fail silently on Windows for non-ASCII paths).

Usage
-----
  python phase2/2_convert_raw_baseline.py \\
      --input-dir data/ground_truth \\
      --out-dir outputs/phase2/raw_baseline

Output
------
  outputs/phase2/raw_baseline/*.png   Same images, format-converted only
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def imread_unicode(path: Path):
    """See 1_preprocess_images.py for full explanation of why this is
    needed instead of cv2.imread() directly (Windows Unicode path bug)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    ext = Path(path).suffix or ".png"
    success, encoded = cv2.imencode(ext, image)
    if not success:
        return False
    encoded.tofile(str(path))
    return True


def main():
    parser = argparse.ArgumentParser(description="Phase 2 — Convert raw images to PNG with NO processing")
    parser.add_argument("--input-dir", type=Path, default=Path("data/ground_truth"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/phase2/raw_baseline"))
    args = parser.parse_args()

    image_files = sorted(args.input_dir.glob("*.jpg"))
    if not image_files:
        raise SystemExit(f"No .jpg images found in {args.input_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    failed = []
    for image_path in tqdm(image_files, desc="Converting (no processing)"):
        image = imread_unicode(image_path)
        if image is None:
            failed.append(image_path.name)
            continue
        out_path = args.out_dir / f"{image_path.stem}.png"
        imwrite_unicode(out_path, image)
        converted += 1

    print(f"\nConverted {converted}/{len(image_files)} images (format-only, no processing).")
    if failed:
        print(f"Failed to read {len(failed)}: {failed}")
    print(f"Saved to: {args.out_dir}")
    print("\nNext step:")
    print(f"  python phase3/1_run_ocr.py --input-dir {args.out_dir} --model tesseract "
          f"--out-root outputs/phase2/raw_baseline_ocr")


if __name__ == "__main__":
    main()
