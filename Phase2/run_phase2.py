"""
phase2/run_phase2.py
━━━━━━━━━━━━━━━━━━━━
Phase 2 Orchestrator — Image Preprocessing

Phase 2 has a single step, so this is a thin wrapper around
1_preprocess_images.py — but it follows the same resumable,
hard-fail-on-error pattern as every other phase's orchestrator for
consistency, and gives a single command to remember regardless of
phase.

Usage
-----
  # Preprocess the Phase 1 ground truth sample (default)
  python phase2/run_phase2.py

  # Preprocess a larger sample instead (e.g. from phase3/0_sample_corpus_for_ocr.py)
  python phase2/run_phase2.py --input-dir data/ocr_sample --out-dir outputs/phase2/preprocessed_sample

  # Force re-processing even if output already exists
  python phase2/run_phase2.py --force
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def run_step(description: str, cmd: list) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {description}")
    print(f"{'=' * 70}")
    print(f"  $ {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"\nStep failed: {description} (exit code {result.returncode})")


def main():
    parser = argparse.ArgumentParser(description="Phase 2 — Run preprocessing pipeline")
    parser.add_argument("--input-dir", type=Path, default=Path("data/ground_truth"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/phase2/preprocessed_images"))
    parser.add_argument("--report", type=Path, default=Path("outputs/phase2/preprocessing_report.csv"))
    parser.add_argument("--comparison-dir", type=Path, default=Path("outputs/phase2/sample_comparisons"))
    parser.add_argument("--num-comparisons", type=int, default=10,
                        help="Number of before/after comparison images (assignment requires "
                             "at least 10)")
    parser.add_argument("--force", action="store_true",
                        help="Re-process even if --out-dir already has output")
    args = parser.parse_args()

    python = sys.executable

    if not args.input_dir.exists():
        sys.exit(
            f"ERROR: {args.input_dir} not found.\n"
            "Run Phase 1 first (python phase1/run_phase1.py), or point --input-dir\n"
            "at a different folder (e.g. data/ocr_sample from phase3/0_sample_corpus_for_ocr.py)."
        )

    already_done = args.out_dir.exists() and any(args.out_dir.glob("*.png"))
    if already_done and not args.force:
        print(f"\n[SKIP] {args.out_dir} already has preprocessed output. Use --force to redo.")
    else:
        run_step(
            "Step 1 — Preprocess images",
            [python, str(SCRIPT_DIR / "1_preprocess_images.py"),
             "--input-dir", str(args.input_dir), "--out-dir", str(args.out_dir),
             "--report", str(args.report), "--comparison-dir", str(args.comparison_dir),
             "--num-comparisons", str(args.num_comparisons)],
        )

    print(f"\n{'=' * 70}")
    print("  Phase 2 complete.")
    print(f"{'=' * 70}")
    print(f"""
Preprocessed images: {args.out_dir}/
Report: {args.report}

Next step:
  python phase3/run_phase3.py
""")


if __name__ == "__main__":
    main()
