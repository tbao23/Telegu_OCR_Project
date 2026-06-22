"""
phase2/3_preprocessing_impact.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 2, Step 3 — Quantify Preprocessing Impact

Merges the "raw baseline" OCR comparison summary against the
"preprocessed" OCR comparison summary (both produced by
phase3/2_compare_ocr_models.py, just pointed at different input
folders) and computes the measured CER/WER delta per model. This is
the actual evidence for "preprocessing contribution is quantified"
(Dimension 5) and "quantitative before/after comparisons demonstrate
measurable improvement" (Dimension 2) — not just a single example page.

Usage
-----
  python phase2/3_preprocessing_impact.py \\
      --raw-summary outputs/phase2/raw_baseline_ocr/metrics/ocr_summary.csv \\
      --preprocessed-summary outputs/phase3_groundtruth/metrics/ocr_summary.csv \\
      --out outputs/phase2/

Output
------
  outputs/phase2/preprocessing_impact.csv
  outputs/phase2/preprocessing_impact.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Phase 2 — Quantify preprocessing's impact on CER/WER")
    parser.add_argument("--raw-summary", type=Path, required=True)
    parser.add_argument("--preprocessed-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/phase2"))
    args = parser.parse_args()

    raw_df = pd.read_csv(args.raw_summary)
    prep_df = pd.read_csv(args.preprocessed_summary)

    # Use the "all_pages" scope row for each model (matches phase3's dual-summary format)
    raw_df = raw_df[raw_df["scope"] == "all_pages"]
    prep_df = prep_df[prep_df["scope"] == "all_pages"]

    merged = raw_df.merge(prep_df, on="model", suffixes=("_raw", "_preprocessed"))

    if merged.empty:
        raise SystemExit(
            "No matching models found between the two summaries. "
            "Check that both were run with the same --models."
        )

    merged["cer_improvement"] = merged["average_cer_raw"] - merged["average_cer_preprocessed"]
    merged["cer_improvement_pct"] = (
        (merged["average_cer_raw"] - merged["average_cer_preprocessed"]) / merged["average_cer_raw"] * 100
    ).round(1)
    merged["wer_improvement"] = merged["average_wer_raw"] - merged["average_wer_preprocessed"]

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "preprocessing_impact.csv"
    merged.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary = []
    for _, row in merged.iterrows():
        direction = "IMPROVED" if row["cer_improvement"] > 0 else "WORSENED"
        summary.append({
            "model": row["model"],
            "raw_cer": round(row["average_cer_raw"], 4),
            "preprocessed_cer": round(row["average_cer_preprocessed"], 4),
            "cer_improvement_pct": row["cer_improvement_pct"],
            "direction": direction,
        })

    json_path = args.out / "preprocessing_impact.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nPreprocessing Impact (raw vs. preprocessed input, same pages, same model):\n")
    for s in summary:
        print(f"  {s['model']}: CER {s['raw_cer']} (raw) -> {s['preprocessed_cer']} (preprocessed) "
              f"= {abs(s['cer_improvement_pct'])}% {s['direction']}")

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
