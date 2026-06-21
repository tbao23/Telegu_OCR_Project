"""
Phase 3 — Run OCR on Preprocessed Telugu Images

Default model:
  Tesseract Telugu OCR

Input:
  phase2/outputs/preprocessed_images/*.png

Output:
  phase3/outputs/tesseract/*.txt
"""
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def run_tesseract(image_path: Path) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        raise SystemExit("Install dependencies: pip install pytesseract pillow")

    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang="tel")
    return text.strip()


def run_easyocr(image_path: Path) -> str:
    try:
        import easyocr
    except ImportError:
        raise SystemExit("Install EasyOCR first: pip install easyocr")

    reader = easyocr.Reader(["te"], gpu=False)
    results = reader.readtext(str(image_path), detail=0)
    return "\n".join(results).strip()


def main():
    parser = argparse.ArgumentParser(description="Phase 3 — Run OCR")
    parser.add_argument("--input-dir", type=Path, default=Path("phase2/outputs/preprocessed_images"))
    parser.add_argument("--model", choices=["tesseract", "easyocr"], default="tesseract")
    parser.add_argument("--out-root", type=Path, default=Path("phase3/outputs"))
    args = parser.parse_args()

    image_files = sorted(args.input_dir.glob("*.png"))

    if not image_files:
        raise SystemExit(f"No .png images found in {args.input_dir}")

    out_dir = args.out_root / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for image_path in tqdm(image_files, desc=f"Running {args.model} OCR"):
        if args.model == "tesseract":
            text = run_tesseract(image_path)
        elif args.model == "easyocr":
            text = run_easyocr(image_path)
        else:
            raise ValueError(f"Unsupported model: {args.model}")

        out_file = out_dir / f"{image_path.stem}.txt"
        out_file.write_text(text, encoding="utf-8")

        records.append({
            "filename": image_path.name,
            "model": args.model,
            "output_file": str(out_file),
            "character_count": len(text),
            "word_count": len(text.split())
        })

    log_path = args.out_root / f"{args.model}_ocr_log.csv"
    pd.DataFrame(records).to_csv(log_path, index=False, encoding="utf-8-sig")

    print("\nPhase 3 OCR complete.")
    print(f"OCR text saved to: {out_dir}")
    print(f"OCR log saved to: {log_path}")


if __name__ == "__main__":
    main()
