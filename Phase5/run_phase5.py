"""
phase5/run_phase5.py
━━━━━━━━━━━━━━━━━━━━
Phase 5 Orchestrator — Runs Analysis Steps in Order

Runs:
  0. Error categorization (substitution/deletion/insertion/diacritic/
     hallucination breakdown)
  1. Scalability and cost estimate for the full corpus

Both steps have no dependency on Ollama or any API — they're pure
local computation, so this orchestrator has no graceful-skip logic
the way run_phase4.py does. It either has the inputs it needs or it
doesn't.

Usage
-----
  python phase5/run_phase5.py \\
      --ground-truth data/ground_truth \\
      --ocr-output data/synthetic_ocr_output/model_a \\
      --model-name model_a \\
      --total-pages 32949
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
    parser = argparse.ArgumentParser(description="Phase 5 — Run analysis pipeline in order")
    parser.add_argument("--ground-truth", type=Path, default=Path("data/ground_truth"))
    parser.add_argument("--ocr-output", type=Path, default=Path("data/synthetic_ocr_output/model_a"))
    parser.add_argument("--model-name", type=str, default="model_a")
    parser.add_argument("--total-pages", type=int, default=32949,
                        help="Total pages in the full corpus, for the cost estimate")
    parser.add_argument("--out", type=Path, default=Path("outputs/phase5"))
    args = parser.parse_args()

    python = sys.executable
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.ocr_output.exists():
        sys.exit(
            f"{args.ocr_output} not found. Run Phase 4 first (python phase4/run_phase4.py) "
            "or point --ocr-output at real OCR results."
        )

    run_step(
        "Step 1 — Error categorization",
        [python, str(SCRIPT_DIR / "1_error_categorization.py"),
         "--ground-truth", str(args.ground_truth), "--ocr-output", str(args.ocr_output),
         "--model-name", args.model_name, "--out", str(args.out)],
    )

    run_step(
        "Step 2 — Scalability and cost estimate",
        [python, str(SCRIPT_DIR / "2_scalability_cost_estimate.py"),
         "--total-pages", str(args.total_pages), "--out", str(args.out)],
    )

    print(f"\n{'=' * 70}")
    print("  Phase 5 automated steps complete.")
    print(f"{'=' * 70}")
    print(f"""
Results saved in: {args.out}/

Remaining manual step:
  Write the final Phase 5 report (model comparison, failure-mode
  discussion, preprocessing impact) pulling in results from:
    - outputs/phase1/  (corpus characterization)
    - outputs/phase4/  (CER/WER, LLM validation, calibration)
    - {args.out}/  (error categories, cost estimate)
""")


if __name__ == "__main__":
    main()
