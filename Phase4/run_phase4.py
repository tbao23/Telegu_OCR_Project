"""
phase4/run_phase4.py
━━━━━━━━━━━━━━━━━━━━
Phase 4 Orchestrator — Runs the Validation Pipeline in Order

Runs the full Phase 4 pipeline end to end:
  0. (Optional) Generate synthetic OCR output for testing
  1. Compute CER/WER (classical metrics)
  2. LLM fluency scoring — Method A (needs Ollama)
  3. LLM error detection — Method B (needs Ollama)
  4. Cross-model agreement — Method C (needs a second model's output)
  5. Calibration analysis (correlates LLM scores against real CER/WER)

Unlike Phase 1, this orchestrator hard-fails fast on any error — including
an unreachable LLM backend. CER/WER (Step 1) does not need the LLM backend
and will have already completed and saved before that check runs, but
Steps 2/3/5 will stop the whole script immediately with a clear error
message rather than silently skipping, so a broken setup can't slip
through unnoticed into your results.

EVERY step also skips itself automatically if its output file already
exists on disk — re-running this script after a partial success will
NOT redo expensive LLM calls you've already paid for/waited on. Pass
--force to override and redo every step regardless.

Usage
-----
  # Default: prompts whether to use synthetic test data, uses local Ollama
  python phase4/run_phase4.py

  # Use a different LLM backend as judge (anthropic/openai/gemini also supported)
  python phase4/run_phase4.py --backend anthropic
  python phase4/run_phase4.py --backend ollama --judge-model qwen3:14b

  # Explicitly use synthetic data (generates it if not already present)
  python phase4/run_phase4.py --use-synthetic

  # Point directly at real Phase 3 output (no synthetic data involved)
  python phase4/run_phase4.py --ocr-output-a data/ocr_output/gemini --model-name-a gemini

  # Skip the cross-model step (only have one model's output)
  python phase4/run_phase4.py --skip-cross-model
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
from llm_backends import check_backend_ready, DEFAULT_MODELS, BACKENDS  # noqa: E402

OUTPUTS_DIR_DEFAULT = Path("outputs/phase4")


def run_step(description: str, cmd: list, hard_fail: bool = True) -> bool:
    """Run a step. Returns True on success. Exits on failure if hard_fail,
    otherwise prints a warning and returns False so the pipeline continues."""
    import subprocess

    print(f"\n{'=' * 70}")
    print(f"  {description}")
    print(f"{'=' * 70}")
    print(f"  $ {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        if hard_fail:
            sys.exit(f"\nStep failed: {description} (exit code {result.returncode})")
        else:
            print(f"\n[SKIPPED] {description} failed (exit code {result.returncode}) — continuing.")
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Phase 4 — Run validation pipeline in order")
    parser.add_argument("--ground-truth", type=Path, default=Path("data/ground_truth"))
    parser.add_argument("--ocr-output-a", type=Path, default=Path("data/synthetic_ocr_output/model_a"))
    parser.add_argument("--ocr-output-b", type=Path, default=Path("data/synthetic_ocr_output/model_b"))
    parser.add_argument("--model-name-a", type=str, default="model_a")
    parser.add_argument("--model-name-b", type=str, default="model_b")
    parser.add_argument("--out", type=Path, default=OUTPUTS_DIR_DEFAULT)
    parser.add_argument("--backend", type=str, default="anthropic", choices=BACKENDS,
                        help="Which LLM backend to use for Methods A/B (default: anthropic — Ollama "
                             "text validation was found to be slow on CPU-only hardware, and Gemini "
                             "had ongoing rate-limit/auth issues; use --backend ollama for fully "
                             "local/free)")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Model name within the chosen backend (default depends on --backend)")
    parser.add_argument("--api-key-env", type=str, default=None,
                        help="Env var name for the API key, overriding the chosen backend's "
                             "standard one (e.g. ANTHROPIC_API_KEY, GOOGLE_API_KEY)")
    parser.add_argument("--error-rate-a", type=float, default=0.08)
    parser.add_argument("--error-rate-b", type=float, default=0.15)
    parser.add_argument("--use-synthetic", action="store_true",
                        help="Generate/use synthetic test data without prompting")
    parser.add_argument("--skip-synthetic", action="store_true",
                        help="Skip synthetic data generation without prompting (use existing --ocr-output-a)")
    parser.add_argument("--skip-cross-model", action="store_true",
                        help="Skip Method C (cross-model agreement) — use if you only have one model's output")
    parser.add_argument("--validation-limit", type=int, default=100,
                        help="Cap LLM fluency/error-detection (Methods A/B) to this many pages "
                             "(default: 100, the assignment's stated minimum). CER/WER and "
                             "cross-model agreement always run on the FULL ground truth set "
                             "regardless, since those are free/local computation — only the "
                             "paid/quota-limited LLM calls are capped. Pass 0 to disable the cap.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run every step even if its output file already exists "
                             "(default: skip steps whose output is already on disk)")
    args = parser.parse_args()

    python = sys.executable
    args.out.mkdir(parents=True, exist_ok=True)
    judge_model = args.judge_model or DEFAULT_MODELS[args.backend]

    if args.use_synthetic and args.skip_synthetic:
        sys.exit("--use-synthetic and --skip-synthetic cannot both be set.")

    # ── Step 0: Synthetic data (optional, with interactive prompt) ─────────────
    a_exists = args.ocr_output_a.exists() and any(args.ocr_output_a.glob("*.txt"))

    if args.skip_synthetic:
        make_synthetic = False
    elif args.use_synthetic:
        make_synthetic = True
    else:
        default_hint = "n" if a_exists else "y"
        if a_exists:
            print(f"\nFound existing OCR output in {args.ocr_output_a} — looks like data may already be ready.")
        answer = input(
            f"Generate synthetic test OCR output now? [y/n] (default: {default_hint}): "
        ).strip().lower() or default_hint
        make_synthetic = answer.startswith("y")

    if make_synthetic:
        run_step(
            "Step 0a — Generate synthetic OCR output (model A)",
            [python, str(SCRIPT_DIR / "0_make_synthetic_test_data.py"),
             "--ground-truth", str(args.ground_truth), "--out", str(args.ocr_output_a),
             "--error-rate", str(args.error_rate_a)],
        )
        if not args.skip_cross_model:
            run_step(
                "Step 0b — Generate synthetic OCR output (model B, for cross-model agreement)",
                [python, str(SCRIPT_DIR / "0_make_synthetic_test_data.py"),
                 "--ground-truth", str(args.ground_truth), "--out", str(args.ocr_output_b),
                 "--error-rate", str(args.error_rate_b), "--seed", "99"],
            )
    else:
        print(f"\nSkipping synthetic data generation. Using existing output at {args.ocr_output_a}")
        if not args.ocr_output_a.exists():
            sys.exit(f"{args.ocr_output_a} does not exist. Re-run and choose 'y', or point --ocr-output-a at real data.")

    # ── Step 1: CER/WER ─────────────────────────────────────────────────────────
    cer_csv = args.out / f"cer_wer_{args.model_name_a}.csv"
    if cer_csv.exists() and not args.force:
        print(f"\n[SKIP] Step 1 — {cer_csv} already exists. Use --force to redo.")
    else:
        run_step(
            "Step 1 — Compute CER/WER (classical metrics)",
            [python, str(SCRIPT_DIR / "1_compute_cer_wer.py"),
             "--ground-truth", str(args.ground_truth), "--ocr-output", str(args.ocr_output_a),
             "--model-name", args.model_name_a, "--out", str(args.out)],
        )

    # ── Steps 2/3/5 need the chosen LLM backend — but only check/fail if at
    #    least one of them actually needs to run (not already skipped below)
    fluency_csv = args.out / f"fluency_{args.model_name_a}.csv"
    error_detect_csv = args.out / f"error_detection_{args.model_name_a}.csv"
    needs_step_2 = args.force or not fluency_csv.exists()
    needs_step_3 = args.force or not error_detect_csv.exists()

    if needs_step_2 or needs_step_3:
        try:
            check_backend_ready(args.backend, api_key_env=args.api_key_env)
        except SystemExit as e:
            sys.exit(
                f"\n{e}\n\n"
                "CER/WER results above were still computed and saved successfully,\n"
                "but the LLM validation steps (fluency scoring, error detection,\n"
                "calibration) cannot proceed until the backend above is ready.\n"
            )

    if not needs_step_2:
        print(f"\n[SKIP] Step 2 — {fluency_csv} already exists. Use --force to redo.")
        fluency_ok = True
    else:
        cmd2 = [python, str(SCRIPT_DIR / "2_llm_fluency_score.py"),
                "--ocr-output", str(args.ocr_output_a), "--model-name", args.model_name_a,
                "--out", str(args.out), "--backend", args.backend, "--judge-model", judge_model]
        if args.api_key_env:
            cmd2 += ["--api-key-env", args.api_key_env]
        if args.validation_limit > 0:
            cmd2 += ["--limit", str(args.validation_limit)]
        fluency_ok = run_step("Step 2 — LLM fluency scoring (Method A)", cmd2)

    if not needs_step_3:
        print(f"\n[SKIP] Step 3 — {error_detect_csv} already exists. Use --force to redo.")
    else:
        cmd3 = [python, str(SCRIPT_DIR / "3_llm_error_detection.py"),
                "--ocr-output", str(args.ocr_output_a), "--model-name", args.model_name_a,
                "--out", str(args.out), "--backend", args.backend, "--judge-model", judge_model]
        if args.api_key_env:
            cmd3 += ["--api-key-env", args.api_key_env]
        if args.validation_limit > 0:
            cmd3 += ["--limit", str(args.validation_limit)]
        run_step("Step 3 — LLM error detection (Method B)", cmd3)

    # ── Step 4: Cross-model agreement (needs model B's output) ─────────────────
    b_exists = args.ocr_output_b.exists() and any(args.ocr_output_b.glob("*.txt"))
    agreement_csv = args.out / "cross_model_agreement.csv"
    if not args.skip_cross_model and agreement_csv.exists() and not args.force:
        print(f"\n[SKIP] Step 4 — {agreement_csv} already exists. Use --force to redo.")
    elif not args.skip_cross_model and b_exists:
        run_step(
            "Step 4 — Cross-model agreement (Method C)",
            [python, str(SCRIPT_DIR / "4_cross_model_agreement.py"),
             "--output-a", str(args.ocr_output_a), "--output-b", str(args.ocr_output_b),
             "--out", str(args.out)],
        )
    elif not args.skip_cross_model:
        sys.exit(
            f"\nERROR: --skip-cross-model not set, but no output found at {args.ocr_output_b}.\n"
            "Either generate/provide a second model's output, or re-run with --skip-cross-model."
        )

    # ── Step 5: Calibration (needs both CER/WER and fluency results) ───────────
    calibration_json = args.out / "calibration_summary.json"
    if calibration_json.exists() and not args.force:
        print(f"\n[SKIP] Step 5 — {calibration_json} already exists. Use --force to redo.")
    else:
        if not (fluency_ok and cer_csv.exists() and fluency_csv.exists()):
            sys.exit(
                "\nERROR: Calibration requires both CER/WER and fluency results, "
                "but one or both are missing. Check the output above for which step failed."
            )
        run_step(
            "Step 5 — Calibration analysis (LLM score vs. real CER)",
            [python, str(SCRIPT_DIR / "5_calibration_analysis.py"),
             "--cer-wer-csv", str(cer_csv), "--fluency-csv", str(fluency_csv),
             "--out", str(args.out)],
        )

    # ── Done ─────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  Phase 4 pipeline complete.")
    print(f"{'=' * 70}")
    print(f"""
Results saved in: {args.out}/

Next step (Phase 5 — error categorization and cost estimate):
  python phase5/run_phase5.py --ground-truth {args.ground_truth} --ocr-output {args.ocr_output_a} --model-name {args.model_name_a}
""")


if __name__ == "__main__":
    main()
