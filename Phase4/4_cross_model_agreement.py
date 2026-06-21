"""
phase4/4_cross_model_agreement.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 4, Step 4 — LLM Validation Method C: Cross-Model Agreement

Compares OCR output from two different models on the same pages. Where
the two outputs diverge significantly, that page is flagged for human
review — high inter-model disagreement predicts low accuracy regions,
without needing ground truth at all.

No API key required — uses difflib's SequenceMatcher, which runs purely
locally. Fully testable today using two synthetic model outputs at
different error rates (see phase4/0_make_synthetic_test_data.py).

Usage
-----
  # First generate two synthetic "models" at different error rates:
  python phase4/0_make_synthetic_test_data.py --out data/synthetic_ocr_output/model_a --error-rate 0.08
  python phase4/0_make_synthetic_test_data.py --out data/synthetic_ocr_output/model_b --error-rate 0.15 --seed 99

  # Then compare them:
  python phase4/4_cross_model_agreement.py \\
      --output-a data/synthetic_ocr_output/model_a \\
      --output-b data/synthetic_ocr_output/model_b \\
      --out outputs/phase4/

Output
------
  outputs/phase4/cross_model_agreement.csv
    One row per page with an agreement_score (0=total disagreement, 1=identical)
"""

import argparse
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


def agreement_score(text_a: str, text_b: str) -> float:
    return SequenceMatcher(None, text_a, text_b).ratio()


def main():
    parser = argparse.ArgumentParser(description="Phase 4 — Cross-model agreement (Method C)")
    parser.add_argument("--output-a", type=Path, required=True, help="Directory with model A's output")
    parser.add_argument("--output-b", type=Path, required=True, help="Directory with model B's output")
    parser.add_argument("--out", type=Path, default=Path("outputs/phase4"))
    parser.add_argument("--flag-threshold", type=float, default=0.7,
                        help="Pages with agreement below this are flagged for review (default 0.7)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    files_a = {f.name for f in args.output_a.glob("*.txt")}
    files_b = {f.name for f in args.output_b.glob("*.txt")}
    common = sorted(files_a & files_b)

    if not common:
        raise SystemExit(
            f"No matching filenames between {args.output_a} and {args.output_b}. "
            "Both directories must use the same page filenames."
        )

    rows = []
    for fname in common:
        text_a = unicodedata.normalize("NFC", (args.output_a / fname).read_text(encoding="utf-8", errors="ignore")).strip()
        text_b = unicodedata.normalize("NFC", (args.output_b / fname).read_text(encoding="utf-8", errors="ignore")).strip()
        if not text_a or not text_b:
            continue
        score = agreement_score(text_a, text_b)
        rows.append({
            "filename": fname,
            "agreement_score": round(score, 4),
            "flagged_for_review": score < args.flag_threshold,
        })

    df = pd.DataFrame(rows)
    csv_path = args.out / "cross_model_agreement.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    n_flagged = df["flagged_for_review"].sum()
    print(f"Compared {len(df)} pages between two models")
    print(f"  Mean agreement: {df['agreement_score'].mean():.4f}")
    print(f"  Flagged for review (agreement < {args.flag_threshold}): {n_flagged} pages")
    print(f"\nSaved: {csv_path}")
    print("\nNext step:")
    print("  python phase4/5_calibration_analysis.py --help")


if __name__ == "__main__":
    main()
