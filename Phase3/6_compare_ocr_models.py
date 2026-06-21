"""
Phase 3 — Compare OCR Output Against Ground Truth

Input:
  data/ground_truth/*.txt
  phase3/outputs/tesseract/*.txt
  phase3/outputs/easyocr/*.txt

Output:
  phase3/outputs/metrics/ocr_comparison_results.csv
  phase3/outputs/metrics/ocr_summary.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def levenshtein_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        return levenshtein_distance(b, a)

    previous_row = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current_row = [i]

        for j, char_b in enumerate(b, start=1):
            insertions = previous_row[j] + 1
            deletions = current_row[j - 1] + 1
            substitutions = previous_row[j - 1] + (char_a != char_b)
            current_row.append(min(insertions, deletions, substitutions))

        previous_row = current_row

    return previous_row[-1]


def character_error_rate(reference: str, prediction: str) -> float:
    if len(reference) == 0:
        return 0.0 if len(prediction) == 0 else 1.0

    distance = levenshtein_distance(reference, prediction)
    return distance / len(reference)


def word_error_rate(reference: str, prediction: str) -> float:
    ref_words = reference.split()
    pred_words = prediction.split()

    if len(ref_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0

    distance = levenshtein_distance(ref_words, pred_words)
    return distance / len(ref_words)


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def compare_model(reference_dir: Path, model_dir: Path, model_name: str) -> list:
    records = []

    reference_files = sorted(reference_dir.glob("*.txt"))

    for ref_path in reference_files:
        prediction_path = model_dir / ref_path.name

        if not prediction_path.exists():
            continue

        reference = normalize_text(ref_path.read_text(encoding="utf-8", errors="ignore"))
        prediction = normalize_text(prediction_path.read_text(encoding="utf-8", errors="ignore"))

        cer = character_error_rate(reference, prediction)
        wer = word_error_rate(reference, prediction)

        records.append({
            "filename": ref_path.name,
            "model": model_name,
            "reference_chars": len(reference),
            "prediction_chars": len(prediction),
            "cer": cer,
            "wer": wer
        })

    return records


def main():
    parser = argparse.ArgumentParser(description="Phase 3 — Compare OCR models")
    parser.add_argument("--reference-dir", type=Path, default=Path("data/ground_truth"))
    parser.add_argument("--ocr-root", type=Path, default=Path("phase3/outputs"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("phase3/outputs/metrics"))
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
        records = compare_model(args.reference_dir, model_dir, model_name)
        all_records.extend(records)

    if not all_records:
        raise SystemExit("No matching OCR outputs found for comparison.")

    results = pd.DataFrame(all_records)

    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    results_path = args.metrics_dir / "ocr_comparison_results.csv"
    summary_path = args.metrics_dir / "ocr_summary.csv"

    results.to_csv(results_path, index=False, encoding="utf-8-sig")

    summary = (
        results.groupby("model")
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

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\nPhase 3 comparison complete.")
    print(f"Detailed results saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")
    print("\nModel Summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
