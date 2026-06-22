"""
phase3/2_compare_ocr_models.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 3, Step 2 — Compare OCR Output Against Ground Truth

Input:
  data/ground_truth/*.txt
  outputs/phase3/tesseract/*.txt
  outputs/phase3/easyocr/*.txt
  outputs/phase3/qwen3vl/*.txt

Output:
  outputs/phase3/metrics/ocr_comparison_results.csv
  outputs/phase3/metrics/ocr_summary.csv

CHANGE LOG (patched):
  - Replaced the original hand-rolled Levenshtein/CER/WER implementation
    with the jiwer library. The assignment explicitly specifies jiwer
    ("Use the jiwer library for CER/WER computation") and the course's
    stated philosophy is that ML engineers should rely on tested
    libraries rather than reimplementing core algorithms from scratch —
    a hand-rolled edit-distance function is exactly the kind of thing
    that should be a library call, not custom code. This also resolves
    having two different CER/WER implementations in the same repo
    (phase4/1_compute_cer_wer.py already used jiwer).
  - normalize_text() applies Unicode NFC normalization in addition to
    whitespace collapsing. Telugu text can represent the same visual
    character via different Unicode code point sequences (precomposed vs.
    decomposed vowel signs/diacritics). Without NFC normalization, CER/WER
    can be artificially inflated by counting normalization differences as
    OCR errors, even when the text is visually/semantically identical.
  - Added --min-ref-chars flagging: pages with very short reference text
    (e.g. near-blank pages) produce statistically unstable CER, since
    CER = edit_distance / reference_length and a tiny denominator can
    blow up the ratio even from modest absolute OCR output. A page with
    23 reference characters and 1000 characters of (otherwise plausible)
    Tesseract output produced a CER of 42.8 on this corpus — one such
    page can dominate a mean-CER summary statistic. compare_model() now
    flags these in the detailed CSV, and the summary is computed BOTH
    including and excluding them, so neither the inflated mean nor a
    silently-filtered number gets reported without context.
"""

import argparse
import unicodedata
from pathlib import Path

import pandas as pd
from jiwer import cer, wer


def normalize_text(text: str) -> str:
    # Apply NFC normalization before whitespace collapsing, so Unicode
    # representation differences (e.g. precomposed vs. decomposed Telugu
    # vowel signs) aren't miscounted as OCR errors.
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.strip().split())


def compare_model(reference_dir: Path, model_dir: Path, model_name: str, min_ref_chars: int = 50) -> list:
    records = []

    reference_files = sorted(reference_dir.glob("*.txt"))

    for ref_path in reference_files:
        prediction_path = model_dir / ref_path.name

        if not prediction_path.exists():
            continue

        reference = normalize_text(ref_path.read_text(encoding="utf-8", errors="ignore"))
        prediction = normalize_text(prediction_path.read_text(encoding="utf-8", errors="ignore"))

        if not reference:
            continue  # no ground truth to compare against

        page_cer = cer(reference, prediction)
        page_wer = wer(reference, prediction)

        records.append({
            "filename": ref_path.name,
            "model": model_name,
            "reference_chars": len(reference),
            "prediction_chars": len(prediction),
            "cer": page_cer,
            "wer": page_wer,
            "flagged_short_reference": len(reference) < min_ref_chars,
        })

    return records


def main():
    parser = argparse.ArgumentParser(description="Phase 3 — Compare OCR models")
    parser.add_argument("--reference-dir", type=Path, default=Path("data/ground_truth"))
    parser.add_argument("--ocr-root", type=Path, default=Path("outputs/phase3"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("outputs/phase3/metrics"))
    parser.add_argument("--min-ref-chars", type=int, default=50,
                        help="Pages with reference text shorter than this are flagged as "
                             "statistically unstable for CER (default: 50)")
    args = parser.parse_args()

    model_dirs = [
        d for d in args.ocr_root.iterdir()
        if d.is_dir() and d.name not in ["metrics", "visualizations"]
    ]

    if not model_dirs:
        raise SystemExit(f"No OCR model folders found in {args.ocr_root}")

    all_records = []

    for model_dir in model_dirs:
        model_name = model_dir.name
        records = compare_model(args.reference_dir, model_dir, model_name, args.min_ref_chars)
        all_records.extend(records)

    if not all_records:
        raise SystemExit("No matching OCR outputs found for comparison.")

    results = pd.DataFrame(all_records)

    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    results_path = args.metrics_dir / "ocr_comparison_results.csv"
    summary_path = args.metrics_dir / "ocr_summary.csv"

    results.to_csv(results_path, index=False, encoding="utf-8-sig")

    def summarize(df: pd.DataFrame, scope: str) -> pd.DataFrame:
        s = (
            df.groupby("model")
            .agg(
                pages_evaluated=("filename", "count"),
                average_cer=("cer", "mean"),
                median_cer=("cer", "median"),
                average_wer=("wer", "mean"),
                median_wer=("wer", "median"),
            )
            .reset_index()
            .sort_values("average_cer")
        )
        s.insert(1, "scope", scope)
        return s

    n_flagged = int(results["flagged_short_reference"].sum())
    summary_all = summarize(results, "all_pages")
    summary_filtered = summarize(
        results[~results["flagged_short_reference"]],
        f"excluding_short_reference (<{args.min_ref_chars} chars, n={n_flagged} flagged)",
    )
    summary = pd.concat([summary_all, summary_filtered], ignore_index=True)

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\nPhase 3 comparison complete.")
    print(f"Detailed results saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")
    if n_flagged > 0:
        print(f"\n{n_flagged} page(s) flagged as short-reference (<{args.min_ref_chars} chars) — "
              "CER on these is statistically unstable (tiny denominator can produce extreme "
              "ratios from modest OCR output). Summary below is reported BOTH ways:")
    print("\nModel Summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
