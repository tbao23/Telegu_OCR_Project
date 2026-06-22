"""
phase4/2_llm_fluency_score.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 4, Step 2 — LLM Validation Method A: Fluency & Coherence Scoring

Uses an LLM to rate the linguistic fluency of OCR output on a 1-5 scale,
WITHOUT needing ground truth — this is the part of the validation
framework that scales to the full corpus. Designed per the assignment's
specified prompt structure.

Backend is configurable via --backend. Ollama (local, free) is the
default; Anthropic/OpenAI/Gemini are available if you'd rather use a
paid API model as the judge. See llm_backends.py (project root) for
setup details for each — including free Gemini API key instructions.

Usage
-----
  # Default: local Ollama, no cost
  python phase4/2_llm_fluency_score.py --ocr-output data/synthetic_ocr_output/model_a --model-name model_a

  # Use Claude instead
  python phase4/2_llm_fluency_score.py --ocr-output ... --model-name model_a --backend anthropic

  # Use a specific model within a backend
  python phase4/2_llm_fluency_score.py --ocr-output ... --model-name model_a --backend ollama --judge-model qwen3:14b

  # Quick test on a few pages first
  python phase4/2_llm_fluency_score.py --ocr-output ... --model-name model_a --limit 5

Output
------
  outputs/phase4/fluency_<model_name>.csv   Per-page score + reasoning
"""

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))  # project root, for llm_backends.py
from llm_backends import call_llm, check_backend_ready, DEFAULT_MODELS, BACKENDS

VALIDATION_PROMPT = """You are an expert in Telugu language and literature.

Below is a passage extracted from a scanned Telugu text via OCR.

Rate the linguistic quality of this text on a scale of 1 to 5:
  5 = Fluent, natural Telugu with no obvious OCR errors
  4 = Mostly fluent with minor errors (1-2 per paragraph)
  3 = Partially readable; some words are garbled
  2 = Significant errors; meaning is often unclear
  1 = Largely unreadable; severe OCR failure

Respond with ONLY a JSON object, no other text, in this exact format:
{{"score": X, "reason": "brief explanation", "error_examples": ["...", "..."]}}

Text:
{ocr_output}"""


def score_page(text: str, backend: str, model: str, api_key_env: str = None, max_retries: int = 3) -> dict:
    prompt = VALIDATION_PROMPT.format(ocr_output=text)
    raw = ""
    for attempt in range(max_retries):
        try:
            raw = call_llm(backend, model, prompt, json_mode=True, max_tokens=700, api_key_env=api_key_env)
            try:
                parsed = json.loads(raw)
                return {
                    "score": parsed.get("score"),
                    "reason": parsed.get("reason", ""),
                    "error_examples": "; ".join(parsed.get("error_examples", [])),
                }
            except json.JSONDecodeError:
                # Response was likely truncated mid-JSON. Salvage the score via
                # regex rather than discarding the whole page's result.
                score_match = re.search(r'"score"\s*:\s*(\d+)', raw)
                reason_match = re.search(r'"reason"\s*:\s*"([^"]*)', raw)
                if score_match:
                    return {
                        "score": int(score_match.group(1)),
                        "reason": (reason_match.group(1) if reason_match else "") + " [response truncated]",
                        "error_examples": "",
                    }
                return {"score": None, "reason": f"PARSE_ERROR: {raw[:200]}", "error_examples": ""}
        except Exception as e:
            if attempt == max_retries - 1:
                return {"score": None, "reason": f"REQUEST_FAILED: {e}", "error_examples": ""}
            time.sleep((2 ** attempt))
    return {"score": None, "reason": "FAILED_AFTER_RETRIES", "error_examples": ""}


def main():
    parser = argparse.ArgumentParser(description="Phase 4 — LLM fluency scoring (Method A)")
    parser.add_argument("--ocr-output", type=Path, required=True,
                        help="Directory with OCR model output .txt files to score")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Label for the OCR model whose output is being scored")
    parser.add_argument("--out", type=Path, default=Path("outputs/phase4"))
    parser.add_argument("--limit", type=int, default=None,
                        help="Only score the first N pages (useful for quick testing)")
    parser.add_argument("--backend", type=str, default="gemini", choices=BACKENDS,
                        help="Which LLM backend to use as the judge (default: gemini — Ollama "
                             "text validation was found to be slow on CPU-only hardware; "
                             "use --backend ollama to go back to fully local/free)")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Model name within the chosen backend (default depends on --backend)")
    parser.add_argument("--api-key-env", type=str, default="GOOGLE_API_KEY_PHASE4",
                        help="Env var name to read the API key from (default: GOOGLE_API_KEY_PHASE4, "
                             "falls back to GOOGLE_API_KEY if not set). Lets Phase 4 use a separate "
                             "Google Cloud project/quota from Phase 3.")
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
        print(f"[{i}/{len(txt_files)}] Scoring {txt_path.name}...")
        result = score_page(text, args.backend, judge_model, api_key_env=args.api_key_env)
        rows.append({"filename": txt_path.name, **result})

    df = pd.DataFrame(rows)
    csv_path = args.out / f"fluency_{args.model_name}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    valid_scores = df["score"].dropna()
    print(f"\nScored {len(df)} pages ({len(valid_scores)} valid)")
    if len(valid_scores) > 0:
        print(f"  Mean fluency score: {valid_scores.mean():.2f} / 5")
    print(f"\nSaved: {csv_path}")
    print("\nNext step:")
    print(f"  python phase4/3_llm_error_detection.py --ocr-output {args.ocr_output} --model-name {args.model_name} --backend {args.backend}")


if __name__ == "__main__":
    main()
