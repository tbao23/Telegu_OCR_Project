"""
phase5/1_error_categorization.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 5 — Error Type Categorization

Goes beyond a single CER number: classifies *what kind* of errors occur
between ground truth and OCR output. The assignment specifically asks
for this in Phase 5 ("error types are categorized and interpreted... in
terms of Telugu script properties").

Categories implemented:
  - Substitution / Deletion / Insertion (via jiwer's alignment)
  - Diacritic (mātrā) errors — substitution where both chars are
    combining vowel signs in the Telugu Unicode block
  - Conjunct-adjacent errors — substitution involving the virama
    (్, U+0C4D) or characters immediately following it
  - Hallucination — hypothesis is >130% the length of reference
    (model invented extra content)
  - Truncation — hypothesis is <70% the length of reference
    (model gave up early / refused / blank)

Fully testable today using synthetic OCR output — no dependency on
Phase 3 deliverables.

Usage
-----
  python phase5/1_error_categorization.py \\
      --ground-truth data/ground_truth \\
      --ocr-output data/synthetic_ocr_output/model_a \\
      --model-name model_a \\
      --out outputs/phase5/

Output
------
  outputs/phase5/error_categories_<model_name>.csv   Per-page breakdown
  outputs/phase5/error_categories_<model_name>_summary.json
"""

import argparse
import json
import unicodedata
from pathlib import Path

import pandas as pd
from jiwer import process_characters

# Telugu combining vowel signs (mātrās) — Unicode block 0C00-0C7F
TELUGU_MATRAS = set("\u0C3E\u0C3F\u0C40\u0C41\u0C42\u0C43\u0C44\u0C46\u0C47\u0C48\u0C4A\u0C4B\u0C4C")
VIRAMA = "\u0C4D"


def load_text(path: Path) -> str:
    return unicodedata.normalize("NFC", path.read_text(encoding="utf-8", errors="ignore")).strip()


def categorize_page(reference: str, hypothesis: str) -> dict:
    counts = {
        "substitutions": 0, "deletions": 0, "insertions": 0,
        "diacritic_errors": 0, "conjunct_adjacent_errors": 0,
    }

    result = process_characters([reference], [hypothesis])
    # jiwer alignment chunks tell us the operation type and the spans involved
    for sentence_chunks in result.alignments:
        for chunk in sentence_chunks:
            if chunk.type == "substitute":
                counts["substitutions"] += (chunk.ref_end_idx - chunk.ref_start_idx)
                ref_span = reference[chunk.ref_start_idx:chunk.ref_end_idx]
                hyp_span = hypothesis[chunk.hyp_start_idx:chunk.hyp_end_idx]
                if any(c in TELUGU_MATRAS for c in ref_span) or any(c in TELUGU_MATRAS for c in hyp_span):
                    counts["diacritic_errors"] += 1
                if VIRAMA in ref_span or VIRAMA in hyp_span:
                    counts["conjunct_adjacent_errors"] += 1
            elif chunk.type == "delete":
                counts["deletions"] += (chunk.ref_end_idx - chunk.ref_start_idx)
            elif chunk.type == "insert":
                counts["insertions"] += (chunk.hyp_end_idx - chunk.hyp_start_idx)

    ref_len = max(len(reference), 1)
    hyp_len = len(hypothesis)
    length_ratio = hyp_len / ref_len

    counts["length_ratio"] = round(length_ratio, 3)
    counts["hallucination_flag"] = length_ratio > 1.3
    counts["truncation_flag"] = length_ratio < 0.7

    return counts


def main():
    parser = argparse.ArgumentParser(description="Phase 5 — Categorize OCR error types")
    parser.add_argument("--ground-truth", type=Path, default=Path("data/ground_truth"))
    parser.add_argument("--ocr-output", type=Path, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/phase5"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for gt_path in sorted(args.ground_truth.glob("*.txt")):
        ocr_path = args.ocr_output / gt_path.name
        reference = load_text(gt_path)
        if not reference or not ocr_path.exists():
            continue
        hypothesis = load_text(ocr_path)
        if not hypothesis:
            continue

        cats = categorize_page(reference, hypothesis)
        rows.append({"filename": gt_path.name, **cats})

    if not rows:
        raise SystemExit("No scoreable page pairs found.")

    df = pd.DataFrame(rows)
    csv_path = args.out / f"error_categories_{args.model_name}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary = {
        "model_name": args.model_name,
        "pages_analyzed": len(df),
        "total_substitutions": int(df["substitutions"].sum()),
        "total_deletions": int(df["deletions"].sum()),
        "total_insertions": int(df["insertions"].sum()),
        "total_diacritic_errors": int(df["diacritic_errors"].sum()),
        "total_conjunct_adjacent_errors": int(df["conjunct_adjacent_errors"].sum()),
        "pages_flagged_hallucination": int(df["hallucination_flag"].sum()),
        "pages_flagged_truncation": int(df["truncation_flag"].sum()),
        "pct_substitution_errors_are_diacritic": round(
            100 * df["diacritic_errors"].sum() / max(df["substitutions"].sum(), 1), 1
        ),
    }
    summary_path = args.out / f"error_categories_{args.model_name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
