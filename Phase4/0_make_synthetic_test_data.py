"""
phase4/0_make_synthetic_test_data.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TEST-ONLY — Synthetic OCR Output Generator

Phase 4 (validation) and Phase 5 (analysis) depend on real OCR output
from Phase 3, which may not exist yet. This script generates a FAKE
"OCR output" folder by deliberately corrupting the Phase 1 ground truth
transcriptions (character substitutions, deletions, insertions at a
controlled error rate) — standing in for a real model's output so the
rest of the Phase 4/5 pipeline can be built, run, and demonstrated
*today*.

DELETE OR IGNORE this script's output once real OCR results from
Phase 3 are available — just point the downstream scripts at the real
output folder instead. The interface (one .txt file per page, same
filename as ground truth) is identical either way, so nothing else
needs to change.

Usage
-----
  python phase4/0_make_synthetic_test_data.py \\
      --ground-truth data/ground_truth \\
      --out data/synthetic_ocr_output/model_a \\
      --error-rate 0.08

  # Generate a second "model" with a different error rate, to test
  # cross-model agreement (Method C) against the first:
  python phase4/0_make_synthetic_test_data.py \\
      --ground-truth data/ground_truth \\
      --out data/synthetic_ocr_output/model_b \\
      --error-rate 0.15 --seed 99
"""

import argparse
import random
import unicodedata
from pathlib import Path

# Telugu Unicode block (for generating plausible substitution noise)
TELUGU_RANGE = (0x0C00, 0x0C7F)


def random_telugu_char() -> str:
    return chr(random.randint(*TELUGU_RANGE))


def corrupt_text(text: str, error_rate: float, rng: random.Random) -> str:
    """
    Apply controlled substitution/deletion/insertion noise to simulate
    OCR errors. error_rate is the approximate fraction of characters
    affected (roughly corresponds to a target CER).
    """
    chars = list(text)
    out = []
    for ch in chars:
        roll = rng.random()
        if roll < error_rate * 0.5:
            # Substitution — replace with a plausible-looking Telugu char
            if ch.strip() and not ch.isspace():
                out.append(random_telugu_char())
            else:
                out.append(ch)
        elif roll < error_rate * 0.75:
            # Deletion — drop the character entirely
            continue
        elif roll < error_rate:
            # Insertion — duplicate-ish noise character before this one
            out.append(random_telugu_char())
            out.append(ch)
        else:
            out.append(ch)
    return "".join(out)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic OCR output for testing Phase 4/5 scripts")
    parser.add_argument("--ground-truth", type=Path, default=Path("data/ground_truth"),
                        help="Directory containing ground-truth .txt files (from Phase 1)")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory for synthetic 'OCR output' .txt files")
    parser.add_argument("--error-rate", type=float, default=0.08,
                        help="Approximate fraction of characters to corrupt (default 0.08 ~ 8%% CER)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(args.ground_truth.glob("*.txt"))
    txt_files = [f for f in txt_files if f.name != "annotation_template.csv"]

    if not txt_files:
        raise SystemExit(
            f"No .txt files found in {args.ground_truth}. "
            "Run phase1/3_sample_ground_truth.py first."
        )

    count = 0
    for txt_path in txt_files:
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue  # skip genuinely empty/untranscribed pages
        corrupted = corrupt_text(text, args.error_rate, rng)
        corrupted = unicodedata.normalize("NFC", corrupted)
        (args.out / txt_path.name).write_text(corrupted, encoding="utf-8")
        count += 1

    print(f"Generated {count} synthetic OCR output files in: {args.out}")
    print(f"Target error rate: ~{args.error_rate:.1%}")
    print("\nNext step:")
    print(f"  python phase4/1_compute_cer_wer.py --ground-truth {args.ground_truth} --ocr-output {args.out}")


if __name__ == "__main__":
    main()
