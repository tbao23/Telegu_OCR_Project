"""
phase4/5_calibration_analysis.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 4, Step 5 — Calibration: Does LLM Validation Reflect True Accuracy?

This is the step the rubric specifically rewards at the top score band:
"results are calibrated against ground truth CER/WER and shown to
correlate meaningfully." It's not enough to compute LLM fluency scores —
you must show whether they actually track real OCR accuracy.

Merges the CER/WER results (1_compute_cer_wer.py) with the LLM fluency
scores (2_llm_fluency_score.py) on filename, computes Pearson and
Spearman correlation, and produces a scatter plot.

Fully testable today with synthetic data — no API key needed for this
script itself (it only reads CSVs already produced by earlier steps).

Usage
-----
  python phase4/5_calibration_analysis.py \\
      --cer-wer-csv outputs/phase4/cer_wer_model_a.csv \\
      --fluency-csv outputs/phase4/fluency_model_a.csv \\
      --out outputs/phase4/

Output
------
  outputs/phase4/calibration_summary.json
  outputs/phase4/calibration_scatter.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def main():
    parser = argparse.ArgumentParser(description="Phase 4 — Calibrate LLM scores against CER/WER")
    parser.add_argument("--cer-wer-csv", type=Path, required=True)
    parser.add_argument("--fluency-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/phase4"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    cer_df = pd.read_csv(args.cer_wer_csv)
    fluency_df = pd.read_csv(args.fluency_csv)

    merged = cer_df.merge(fluency_df, on="filename", how="inner")
    merged = merged.dropna(subset=["cer", "score"])

    n_total = len(merged)
    n_flagged = 0
    if "flagged_short_reference" in merged.columns:
        n_flagged = int(merged["flagged_short_reference"].sum())
        merged = merged[~merged["flagged_short_reference"]]
        if n_flagged > 0:
            print(f"Excluded {n_flagged} short-reference page(s) (statistically unstable CER) "
                  f"before computing correlation — {n_total} pages available, {len(merged)} used.")

    if len(merged) < 3:
        raise SystemExit(
            f"Only {len(merged)} pages have both CER and a valid LLM score (after excluding "
            f"{n_flagged} short-reference outlier(s)) — need at least 3 to compute meaningful correlation."
        )

    # Fluency score is 1 (worst) to 5 (best); CER is 0 (best) to 1+ (worst).
    # We expect a NEGATIVE correlation: higher fluency score = lower CER.
    pearson_r, pearson_p = pearsonr(merged["score"], merged["cer"])
    spearman_r, spearman_p = spearmanr(merged["score"], merged["cer"])

    summary = {
        "n_pages": len(merged),
        "n_pages_excluded_short_reference": n_flagged,
        "pearson_r": round(float(pearson_r), 4),
        "pearson_p_value": round(float(pearson_p), 4),
        "spearman_r": round(float(spearman_r), 4),
        "spearman_p_value": round(float(spearman_p), 4),
        "interpretation": (
            "Strong negative correlation expected (fluency score should drop as CER rises). "
            f"Observed Pearson r = {pearson_r:.3f}. "
            + ("This suggests the LLM fluency score IS a meaningful proxy for OCR accuracy."
               if pearson_r < -0.4 and pearson_p < 0.05
               else "This is weaker than expected — the LLM score may not reliably track "
                    "true OCR accuracy on this sample; consider a larger sample or prompt revision.")
        ),
    }

    summary_path = args.out / "calibration_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Scatter plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(merged["cer"], merged["score"], alpha=0.7, color="#2980b9", edgecolor="white")
    ax.set_xlabel("Character Error Rate (CER) — lower is better")
    ax.set_ylabel("LLM Fluency Score (1-5) — higher is better")
    ax.set_title(f"LLM Fluency Score vs. CER (Pearson r = {pearson_r:.3f}, p = {pearson_p:.3f})")
    plt.tight_layout()
    plot_path = args.out / "calibration_scatter.png"
    fig.savefig(plot_path, dpi=130)
    plt.close()

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
