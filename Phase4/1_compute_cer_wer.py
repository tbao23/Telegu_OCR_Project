"""
phase4/1_compute_cer_wer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 4, Step 1 — Classical Evaluation Metrics (CER / WER)

Computes Character Error Rate and Word Error Rate for each page in the
ground truth sample against the corresponding OCR model output, using
the jiwer library. Produces a per-page CSV and an aggregate summary.

Works identically whether --ocr-output points at synthetic test data
(phase4/0_make_synthetic_test_data.py) or real Phase 3 model output —
the interface is one .txt file per page, matched by filename.

CHANGE LOG (patched):
  - Added --min-ref-chars flagging, matching phase3/2_compare_ocr_models.py.
    CER = edit_distance / reference_length, so a page with a tiny
    reference (e.g. a near-blank page) can produce an extreme CER from
    even modest OCR output — one such page (23 reference chars, CER
    42.8) dominated this corpus's mean-CER statistic before this fix.
    Summary is now reported BOTH with and without flagged pages, so
    neither number gets used without that context — and critically,
    this CSV feeds 5_calibration_analysis.py's correlation calculation,
    where a single extreme outlier can distort the correlation result
    enough to misrepresent whether the LLM judge actually tracks
    real accuracy.

Usage
-----
  python phase4/1_compute_cer_wer.py \\
      --ground-truth data/ground_truth \\
      --ocr-output data/synthetic_ocr_output/model_a \\
      --model-name model_a \\
      --out outputs/phase4/

Output
------
  outputs/phase4/cer_wer_<model_name>.csv      Per-page CER/WER
  outputs/phase4/cer_wer_<model_name>_summary.json   Aggregate stats (both ways)
"""

import argparse
import json
import unicodedata
from pathlib import Path

import pandas as pd
from jiwer import cer, wer


def load_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return unicodedata.normalize("NFC", text).strip()


def main():
    parser = argparse.ArgumentParser(description="Phase 4 — Compute CER/WER against ground truth")
    parser.add_argument("--ground-truth", type=Path, default=Path("data/ground_truth"),
                        help="Directory with ground-truth .txt files")
    parser.add_argument("--ocr-output", type=Path, required=True,
                        help="Directory with OCR model output .txt files (same filenames as ground truth)")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Label for this model/run, e.g. 'gemini-1.5-pro' or 'model_a'")
    parser.add_argument("--out", type=Path, default=Path("outputs/phase4"),
                        help="Output directory for results")
    parser.add_argument("--min-ref-chars", type=int, default=50,
                        help="Pages with reference text shorter than this are flagged as "
                             "statistically unstable for CER (default: 50)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    gt_files = sorted(args.ground_truth.glob("*.txt"))
    rows = []
    skipped = []

    for gt_path in gt_files:
        ocr_path = args.ocr_output / gt_path.name
        reference = load_text(gt_path)

        if not reference:
            skipped.append((gt_path.name, "empty ground truth (no transcription available)"))
            continue

        if not ocr_path.exists():
            skipped.append((gt_path.name, "no matching OCR output file"))
            continue

        hypothesis = load_text(ocr_path)
        if not hypothesis:
            skipped.append((gt_path.name, "empty OCR output"))
            continue

        page_cer = cer(reference, hypothesis)
        page_wer = wer(reference, hypothesis)

        rows.append({
            "filename": gt_path.name,
            "ref_chars": len(reference),
            "hyp_chars": len(hypothesis),
            "cer": round(page_cer, 4),
            "wer": round(page_wer, 4),
            "flagged_short_reference": len(reference) < args.min_ref_chars,
        })

    if not rows:
        raise SystemExit(
            "No page pairs could be scored. Check that --ocr-output contains "
            "files matching the ground-truth filenames."
        )

    df = pd.DataFrame(rows)
    csv_path = args.out / f"cer_wer_{args.model_name}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    n_flagged = int(df["flagged_short_reference"].sum())
    df_filtered = df[~df["flagged_short_reference"]]

    def stats_block(d: pd.DataFrame) -> dict:
        if len(d) == 0:
            return {"pages": 0, "mean_cer": None, "median_cer": None, "mean_wer": None, "median_wer": None}
        return {
            "pages": len(d),
            "mean_cer": round(float(d["cer"].mean()), 4),
            "median_cer": round(float(d["cer"].median()), 4),
            "mean_wer": round(float(d["wer"].mean()), 4),
            "median_wer": round(float(d["wer"].median()), 4),
        }

    summary = {
        "model_name": args.model_name,
        "pages_scored": len(df),
        "pages_skipped": len(skipped),
        "skip_reasons": skipped,
        "pages_flagged_short_reference": n_flagged,
        "all_pages": stats_block(df),
        f"excluding_short_reference_lt_{args.min_ref_chars}_chars": stats_block(df_filtered),
        "worst_5_pages_by_cer": df.nlargest(5, "cer")[["filename", "ref_chars", "cer", "wer"]].to_dict("records"),
        "best_5_pages_by_cer": df.nsmallest(5, "cer")[["filename", "ref_chars", "cer", "wer"]].to_dict("records"),
    }
    summary_path = args.out / f"cer_wer_{args.model_name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nScored {len(df)} pages for model '{args.model_name}' ({len(skipped)} skipped)")
    print(f"  All pages       — Mean CER: {summary['all_pages']['mean_cer']}   Median CER: {summary['all_pages']['median_cer']}")
    if n_flagged > 0:
        excl_key = f"excluding_short_reference_lt_{args.min_ref_chars}_chars"
        print(f"  Excl. {n_flagged} short-ref — Mean CER: {summary[excl_key]['mean_cer']}   Median CER: {summary[excl_key]['median_cer']}")
        print(f"  ({n_flagged} page(s) had reference text under {args.min_ref_chars} chars — see summary JSON for details)")
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {summary_path}")
    print("\nNext step:")
    print(f"  python phase4/2_llm_fluency_score.py --ocr-output {args.ocr_output} --model-name {args.model_name}")


if __name__ == "__main__":
    main()
