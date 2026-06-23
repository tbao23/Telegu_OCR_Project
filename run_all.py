"""
run_all.py  (project root)
━━━━━━━━━━━━━━━━━━━━━━━━━━
Master Orchestrator — Runs Phases 1 through 5 End to End

A single command to reproduce the entire project from a fresh state
(corpus already downloaded — this script does NOT re-download the
~10.3 GB corpus, since that doesn't change between runs).

Sequence
--------
  1. Phase 1 — prompts whether to create a NEW ground truth sample.
       y: runs the full Phase 1 (profile, characterize, fresh random
          sample) — you'll need to annotate the new sample manually.
       n: skips Phase 1 ENTIRELY — your existing data/ground_truth/
          (including your own completed annotation) is left untouched.
  2. Phase 2 — preprocess whatever's in data/ground_truth/ (new or kept)
  2b. Preprocessing Impact — runs OCR on the SAME 40 ground-truth pages
      both raw (no processing) and preprocessed, producing a real
      measured CER delta (not just a single anecdotal example) — this
      satisfies Dimension 2 and 5's requirement to quantify preprocessing's
      actual contribution to accuracy.
  3. Phase 3 — sample N pages from the full corpus, preprocess, run OCR
     with both models, compare against reference text
  4. Phase 4 — CER/WER, LLM validation (capped at --validation-limit
     pages), cross-model agreement, calibration — on REAL Phase 3 output
  5. Phase 5 — error categorization + scalability/cost estimate — on
     the same real data

Every phase's own orchestrator is resumable (skips steps whose output
already exists) — so if this script is interrupted (e.g. a quota issue
that exhausts every rotation key), just re-run the SAME command and it
picks up where it left off rather than starting over.

Usage
-----
  # Default: asks whether to create a new ground truth sample
  python run_all.py

  # Skip the prompt — explicitly keep existing ground truth as-is
  python run_all.py --keep-ground-truth

  # Skip the prompt — explicitly create a fresh random sample
  python run_all.py --new-ground-truth-sample
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent


def run_step(description: str, cmd: list) -> None:
    print(f"\n{'#' * 70}")
    print(f"#  {description}")
    print(f"{'#' * 70}")
    print(f"  $ {' '.join(str(c) for c in cmd)}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(
            f"\n{'!' * 70}\n"
            f"PIPELINE STOPPED: {description} failed (exit code {result.returncode}).\n"
            f"Fix the issue above, then re-run this SAME command — every phase\n"
            f"is resumable, so completed steps will be skipped automatically.\n"
            f"{'!' * 70}"
        )


def main():
    parser = argparse.ArgumentParser(description="Run all 5 phases end to end")
    parser.add_argument("--keep-ground-truth", action="store_true",
                        help="Skip the prompt — explicitly keep data/ground_truth/ as-is, "
                             "skipping Phase 1 entirely")
    parser.add_argument("--new-ground-truth-sample", action="store_true",
                        help="Skip the prompt — explicitly run Phase 1 fresh (new random sample)")
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    parser.add_argument("--sample-size", type=int, default=500,
                        help="Pages to sample from the full corpus for Phase 3 (default: 500, "
                             "the final-deliverable minimum)")
    parser.add_argument("--validation-limit", type=int, default=100,
                        help="Pages to use for Phase 4's LLM validation (default: 100, the "
                             "assignment's stated minimum — kept separate from --sample-size "
                             "since using all 500 there would waste API quota for no rubric benefit)")
    parser.add_argument("--models", type=str, default="claude,tesseract",
                        help="OCR models to run in Phase 3 (default: claude,tesseract — claude "
                             "first/primary since it's the model validated in Phase 4/5; Gemini "
                             "abandoned due to ongoing rate-limit/auth issues with its new key "
                             "format, see llm_backends.py)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run every step regardless of existing output (passed through to "
                             "every phase, including Phase 1 if a new sample is being created)")
    args = parser.parse_args()

    if args.keep_ground_truth and args.new_ground_truth_sample:
        sys.exit("--keep-ground-truth and --new-ground-truth-sample cannot both be set.")

    if not args.corpus_dir.exists():
        sys.exit(
            f"ERROR: {args.corpus_dir} not found. This script does not download the corpus "
            "(it doesn't change between runs) — run phase1/0_download_corpus.py first."
        )

    python = sys.executable
    force_flag = ["--force"] if args.force else []

    # ── Phase 1 — single decision point ─────────────────────────────────────
    gt_dir = Path("data/ground_truth")
    if args.keep_ground_truth:
        create_new = False
    elif args.new_ground_truth_sample:
        create_new = True
    else:
        existing = gt_dir.exists() and (gt_dir / "annotation_template.csv").exists()
        default_hint = "n" if existing else "y"
        if existing:
            print(f"\nFound existing annotated ground truth in {gt_dir}.")
        answer = input(
            f"Create a NEW ground truth sample for Phase 1? [y/n] (default: {default_hint}): "
        ).strip().lower() or default_hint
        create_new = answer.startswith("y")

    if create_new:
        run_step(
            "Phase 1 — Run (profile, characterize, sample ground truth)",
            [python, "phase1/run_phase1.py", "--skip-download", "--corpus-dir", str(args.corpus_dir)] + force_flag,
        )
        print("\n[NOTICE] A fresh random ground truth sample was created — you'll need to")
        print("         manually annotate data/ground_truth/annotation_template.csv before")
        print("         the Phase 1 deliverable is complete (this script doesn't do that part).")
    else:
        print(f"\n[SKIP] Phase 1 — keeping existing {gt_dir} untouched.")
        if not (gt_dir / "annotation_template.csv").exists():
            sys.exit(
                f"ERROR: {gt_dir}/annotation_template.csv doesn't exist, so there's nothing "
                "to keep. Re-run and choose 'y', or restore a backup first with "
                "phase1/optional_restore_ground_truth.py."
            )

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    run_step(
        "Phase 2 — Preprocess ground truth images",
        [python, "phase2/run_phase2.py"] + force_flag,
    )

    # ── Preprocessing Impact Quantification ─────────────────────────────────
    # Required by Dimension 2 ("quantitative before/after comparisons
    # demonstrate measurable improvement") and Dimension 5 ("preprocessing
    # contribution is quantified") — runs OCR on the SAME 40 ground-truth
    # pages both raw and preprocessed, for a real controlled comparison.
    model_names = [m.strip() for m in args.models.split(",")]
    raw_baseline_dir = Path("outputs/phase2/raw_baseline")
    raw_ocr_root = Path("outputs/phase2/raw_baseline_ocr")
    prep_ocr_root = Path("outputs/phase2/preprocessed_ocr")

    convert_log = raw_baseline_dir
    if convert_log.exists() and any(convert_log.glob("*.png")) and not args.force:
        print(f"\n[SKIP] Raw baseline conversion — {raw_baseline_dir} already populated. Use --force to redo.")
    else:
        run_step(
            "Preprocessing Impact — Convert raw baseline (no processing)",
            [python, "phase2/2_convert_raw_baseline.py",
             "--input-dir", "data/ground_truth", "--out-dir", str(raw_baseline_dir)],
        )

    for model in model_names:
        for label, input_dir, out_root in [
            ("raw", raw_baseline_dir, raw_ocr_root),
            ("preprocessed", Path("outputs/phase2/preprocessed_images"), prep_ocr_root),
        ]:
            log_path = out_root / f"{model}_ocr_log.csv"
            if log_path.exists() and not args.force:
                print(f"\n[SKIP] Preprocessing Impact — {label} OCR ({model}) already done. Use --force to redo.")
                continue
            cmd = [python, "phase3/1_run_ocr.py",
                   "--input-dir", str(input_dir), "--model", model, "--out-root", str(out_root)]
            if model == "gemini":
                cmd += ["--api-key-env", "GOOGLE_API_KEY_PHASE3"]
            run_step(f"Preprocessing Impact — Run {label} OCR ({model})", cmd)

    for label, ocr_root in [("raw", raw_ocr_root), ("preprocessed", prep_ocr_root)]:
        summary_path = ocr_root / "metrics" / "ocr_summary.csv"
        if summary_path.exists() and not args.force:
            print(f"\n[SKIP] Preprocessing Impact — {label} comparison already done. Use --force to redo.")
            continue
        run_step(
            f"Preprocessing Impact — Compare {label} OCR against ground truth",
            [python, "phase3/2_compare_ocr_models.py",
             "--reference-dir", "data/ground_truth", "--ocr-root", str(ocr_root),
             "--metrics-dir", str(ocr_root / "metrics")],
        )

    impact_json = Path("outputs/phase2/preprocessing_impact.json")
    if impact_json.exists() and not args.force:
        print(f"\n[SKIP] Preprocessing Impact — {impact_json} already exists. Use --force to redo.")
    else:
        run_step(
            "Preprocessing Impact — Compute raw-vs-preprocessed CER delta",
            [python, "phase2/3_preprocessing_impact.py",
             "--raw-summary", str(raw_ocr_root / "metrics" / "ocr_summary.csv"),
             "--preprocessed-summary", str(prep_ocr_root / "metrics" / "ocr_summary.csv"),
             "--out", "outputs/phase2"],
        )

    # ── Phase 3 ──────────────────────────────────────────────────────────────
    run_step(
        f"Phase 3 — Sample {args.sample_size} pages, preprocess, OCR ({args.models}), compare",
        [python, "phase3/run_phase3.py", "--use-corpus-sample",
         "--sample-size", str(args.sample_size), "--models", args.models,
         "--corpus-dir", str(args.corpus_dir)] + force_flag,
    )

    # ── Phase 4 ──────────────────────────────────────────────────────────────
    model_a, model_b = model_names[0], (model_names[1] if len(model_names) > 1 else model_names[0])
    run_step(
        "Phase 4 — CER/WER, LLM validation, cross-model agreement, calibration (real data)",
        [python, "phase4/run_phase4.py",
         "--ground-truth", "data/ocr_sample",
         "--ocr-output-a", f"outputs/phase3/{model_a}", "--model-name-a", model_a,
         "--ocr-output-b", f"outputs/phase3/{model_b}", "--model-name-b", model_b,
         "--skip-synthetic", "--validation-limit", str(args.validation_limit)] + force_flag,
    )

    # ── Phase 5 ──────────────────────────────────────────────────────────────
    run_step(
        "Phase 5 — Error categorization + scalability/cost estimate (real data)",
        [python, "phase5/run_phase5.py",
         "--ground-truth", "data/ocr_sample",
         "--ocr-output", f"outputs/phase3/{model_a}", "--model-name", model_a,
         "--total-pages", "32949"] + force_flag,
    )

    # ── Done ─────────────────────────────────────────────────────────────────
    print(f"\n{'#' * 70}")
    print("#  ALL 5 PHASES COMPLETE")
    print(f"{'#' * 70}")
    print("""
Results:
  outputs/                  Phase 1 corpus characterization
  data/ground_truth/        Restored 40-page ground truth + annotation
  outputs/phase2/           Preprocessed images + preprocessing_impact.csv/json
  data/ocr_sample/          500-page corpus sample (Phase 3 input)
  outputs/phase3/           OCR output (both models) + comparison metrics
  outputs/phase4/           CER/WER, LLM validation, calibration results
  outputs/phase5/           Error categories, cost estimate

Remaining manual steps:
  1. Write/finalize the final report (all 5 phases, 15+ pages, Quarto -> PDF)
  2. Record the ~10 min presentation
  3. Rename the project folder to FinalProject_[LastName]_[FirstName]
  4. Share with read access by the deadline
""")


if __name__ == "__main__":
    main()
