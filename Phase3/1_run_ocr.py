"""
phase3/1_run_ocr.py
━━━━━━━━━━━━━━━━━━━━
Phase 3, Step 1 — Run OCR on Preprocessed Telugu Images

Models:
  tesseract  Tesseract Telugu OCR (free baseline)
  easyocr    EasyOCR Telugu model
  qwen3vl    Qwen3-VL vision-language model via local Ollama (free, no API key)
  gemini     Gemini vision model via Google AI Studio (FREE TIER, no credit
             card — see llm_backends.py at project root for setup. The
             assignment specifically calls out Gemini as "Excellent;
             top-performing on Indic scripts.")

Input:
  outputs/phase2/preprocessed_images/*.png

Output:
  outputs/phase3/<model>/*.txt

CHANGE LOG (patched):
  - tesseract_cmd is no longer hardcoded at module level. It's now only
    set if --tesseract-path is passed or the TESSERACT_PATH env var is
    set; otherwise pytesseract looks it up on PATH as normal.
  - qwen3vl and gemini both route through the shared llm_backends.py
    module (project root) instead of duplicating backend-specific HTTP
    code here — same module Phase 4 uses for text validation, just with
    an image attached.
"""
import argparse
import os
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))  # project root, for llm_backends.py
from llm_backends import call_llm, check_backend_ready, DEFAULT_VISION_MODELS

# Maps the --model CLI choice to the backend name llm_backends.py expects
VISION_MODEL_TO_BACKEND = {
    "qwen3vl": "ollama",
    "gemini": "gemini",
}

# Per the assignment's recommended prompt structure for vision-LLM Telugu OCR
OCR_SYSTEM_PROMPT = """You are an expert OCR assistant specializing in Telugu script.

Your task is to extract all text from the provided image, exactly as it appears.

Rules:
- Output only the recognized Telugu text in Unicode (UTF-8).
- Preserve line breaks as they appear in the original.
- Do not translate, interpret, or summarize.
- If a character is ambiguous, output your best guess and mark it with [?].
- Do not output any English explanation."""


def run_tesseract(image_path: Path, tesseract_path: str = None) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        raise SystemExit("Install dependencies: pip install pytesseract pillow")

    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

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


def run_vision_ocr(image_path: Path, backend: str, model: str, api_key_env: str = None, max_retries: int = 5) -> str:
    """OCR a page image using any vision-capable backend via llm_backends.py."""
    for attempt in range(max_retries):
        try:
            text = call_llm(backend, model, OCR_SYSTEM_PROMPT, image_path=image_path,
                             max_tokens=2000, api_key_env=api_key_env)
            return unicodedata.normalize("NFC", text.strip())
        except Exception as e:
            if attempt == max_retries - 1:
                return f"[OCR_FAILED: {e}]"
            wait = (2 ** attempt) + 0.5
            print(f"  Retry {attempt + 1}/{max_retries} after {wait:.1f}s: {e}")
            time.sleep(wait)
    return "[OCR_FAILED: max retries exceeded]"


def main():
    parser = argparse.ArgumentParser(description="Phase 3 — Run OCR")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/phase2/preprocessed_images"))
    parser.add_argument("--model", choices=["tesseract", "easyocr", "qwen3vl", "gemini"], default="tesseract")
    parser.add_argument("--out-root", type=Path, default=Path("outputs/phase3"))
    parser.add_argument("--tesseract-path", type=str, default=os.environ.get("TESSERACT_PATH"),
                        help="Path to tesseract.exe, if not on PATH. Can also set TESSERACT_PATH env var.")
    parser.add_argument("--vision-model", type=str, default=None,
                        help="Specific model name for qwen3vl/gemini (default depends on --model)")
    parser.add_argument("--api-key-env", type=str, default="GOOGLE_API_KEY_PHASE3",
                        help="Env var name to read the API key from (default: GOOGLE_API_KEY_PHASE3, "
                             "falls back to GOOGLE_API_KEY if not set). Lets Phase 3 use a separate "
                             "Google Cloud project/quota from Phase 4.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N images (useful for quick testing)")
    args = parser.parse_args()

    image_files = sorted(args.input_dir.glob("*.png"))

    if not image_files:
        raise SystemExit(f"No .png images found in {args.input_dir}")

    if args.limit:
        image_files = image_files[:args.limit]

    backend = None
    vision_model = None
    if args.model in VISION_MODEL_TO_BACKEND:
        backend = VISION_MODEL_TO_BACKEND[args.model]
        vision_model = args.vision_model or DEFAULT_VISION_MODELS[backend]
        check_backend_ready(backend, api_key_env=args.api_key_env)
        print(f"Vision backend: {backend}   Model: {vision_model}   Key env: {args.api_key_env}\n")

    out_dir = args.out_root / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for image_path in tqdm(image_files, desc=f"Running {args.model} OCR"):
        if args.model == "tesseract":
            text = run_tesseract(image_path, tesseract_path=args.tesseract_path)
        elif args.model == "easyocr":
            text = run_easyocr(image_path)
        elif args.model in VISION_MODEL_TO_BACKEND:
            text = run_vision_ocr(image_path, backend, vision_model, api_key_env=args.api_key_env)
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
    print("\nNext step (point --reference-dir at whichever folder has the matching .txt reference files):")
    print(f"  python phase3/2_compare_ocr_models.py --reference-dir data/ocr_sample --ocr-root {args.out_root}")


if __name__ == "__main__":
    main()
