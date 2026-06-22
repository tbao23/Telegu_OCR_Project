"""
phase4/3_llm_error_detection.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 4, Step 3 — LLM Validation Method B: Error Detection & Correction

Uses an LLM to identify likely OCR errors by exploiting its knowledge of
Telugu morphology and vocabulary, and suggest corrections. Like Method A,
this requires no ground truth and scales to the full corpus.

Backend is configurable via --backend (ollama/anthropic/openai/gemini).
See llm_backends.py (project root) for setup details for each,
including free Gemini API key instructions. Ollama (local, free) is
the default.

Usage
-----
  python phase4/3_llm_error_detection.py \\
      --ocr-output data/synthetic_ocr_output/model_a \\
      --model-name model_a \\
      --backend ollama

Output
------
  outputs/phase4/error_detection_<model_name>.csv
    One row per page with: filename, num_flagged_errors, raw_response
"""

import argparse
import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))  # project root, for llm_backends.py
from llm_backends import call_llm, check_backend_ready, DEFAULT_MODELS, BACKENDS

ERROR_DETECTION_PROMPT = """You are a Telugu language expert and proofreader.

The following text was produced by an OCR system from a scanned Telugu document.

Identify words or sequences that are likely OCR errors (not valid Telugu words or
morphologically implausible), and suggest the most probable correction.

Format each finding on its own line exactly as:
[ERROR] -> [CORRECTION] (reason)

If you find no errors, respond with exactly: NO_ERRORS_DETECTED

Text:
{ocr_output}"""


def detect_errors(text: str, backend: str, model: str, api_key_env: str = None, max_retries: int = 3) -> dict:
    prompt = ERROR_DETECTION_PROMPT.format(ocr_output=text)
    for attempt in range(max_retries):
        try:
            raw = call_llm(backend, model, prompt, json_mode=False, max_tokens=700, api_key_env=api_key_env)
            if raw == "NO_ERRORS_DETECTED" or "NO_ERRORS_DETECTED" in raw:
                return {"num_flagged_errors": 0, "raw_response": raw}
            findings = re.findall(r".+->.+", raw)
            return {"num_flagged_errors": len(findings), "raw_response": raw}
        except Exception as e:
            if attempt == max_retries - 1:
                return {"num_flagged_errors": None, "raw_response": f"REQUEST_FAILED: {e}"}
            time.sleep((2 ** attempt))
    return {"num_flagged_errors": None, "raw_response": "FAILED_AFTER_RETRIES"}


def main():
    parser = argparse.ArgumentParser(description="Phase 4 — LLM error detection (Method B)")
    parser.add_argument("--ocr-output", type=Path, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/phase4"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backend", type=str, default="gemini", choices=BACKENDS,
                        help="Which LLM backend to use as the judge (default: gemini — see "
                             "2_llm_fluency_score.py for why; use --backend ollama for fully local/free)")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Model name within the chosen backend (default depends on --backend)")
    parser.add_argument("--api-key-env", type=str, default="GOOGLE_API_KEY_PHASE4",
                        help="Env var name to read the API key from (default: GOOGLE_API_KEY_PHASE4, "
                             "falls back to GOOGLE_API_KEY if not set).")
    args = parser.parse_args()

    judge_model = args.judge_model or DEFAULT_MODELS[args.backend]
    check_backend_ready(args.backend, api_key_env=args.api_key_env)
    args.out.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(args.ocr_output.glob("*.txt"))
    if args.limit:
        txt_files = txt_files[:args.limit]

    if not txt_files:
        raise SystemExit(f"No .txt files found in {args.ocr_output}")

    print(f"Backend: {args.backend}   Model: {judge_model}\n")

    rows = []
    for i, txt_path in enumerate(txt_files, 1):
        text = unicodedata.normalize("NFC", txt_path.read_text(encoding="utf-8", errors="ignore")).strip()
        if not text:
            continue
        print(f"[{i}/{len(txt_files)}] Checking {txt_path.name}...")
        result = detect_errors(text, args.backend, judge_model, api_key_env=args.api_key_env)
        rows.append({"filename": txt_path.name, **result})

    df = pd.DataFrame(rows)
    csv_path = args.out / f"error_detection_{args.model_name}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    valid = df["num_flagged_errors"].dropna()
    print(f"\nChecked {len(df)} pages")
    if len(valid) > 0:
        print(f"  Mean flagged errors per page: {valid.mean():.2f}")
    print(f"\nSaved: {csv_path}")
    print("\nNext step:")
    print("  Once you have output from a SECOND model, run:")
    print(f"  python phase4/4_cross_model_agreement.py --output-a {args.ocr_output} --output-b <second_model_dir>")


if __name__ == "__main__":
    main()
