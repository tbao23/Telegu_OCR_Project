"""
phase3/run_phase3.py
━━━━━━━━━━━━━━━━━━━━
Phase 3 Orchestrator — OCR Pipeline and Model Comparison

Runs the full Phase 3 pipeline end to end:
  0. (Optional) Sample additional pages from the full corpus
  1. Preprocess images (calls phase2/1_preprocess_images.py — OCR needs
     preprocessed input, so this orchestrator pulls it in directly
     rather than requiring a separate manual step)
  2. Run OCR for each requested model (tesseract, qwen3vl, easyocr)
  3. Compare all models against the reference text

Like run_phase4.py, this hard-fails immediately and clearly if a required
backend isn't ready (e.g. Ollama not running for qwen3vl) — it does not
silently skip. Every step is resumable: it's skipped if its output
already exists, unless --force is passed.

Usage
-----
  # Default: use the existing 40-page Phase 1 ground truth, run
  # tesseract + qwen3vl, compare results
  python phase3/run_phase3.py

  # Sample 100 pages from the full corpus first, then run on those instead
  python phase3/run_phase3.py --use-corpus-sample --sample-size 100

  # Only run Tesseract (skip the Ollama-dependent vision model)
  python phase3/run_phase3.py --models tesseract

  # Redo everything regardless of existing output
  python phase3/run_phase3.py --force
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PHASE2_DIR = SCRIPT_DIR.parent / "phase2"
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
from llm_backends import check_backend_ready  # noqa: E402  (reused for the Ollama readiness check)


def run_step(description: str, cmd: list) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {description}")
    print(f"{'=' * 70}")
    print(f"  $ {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"\nStep failed: {description} (exit code {result.returncode})")


def main():
    parser = argparse.ArgumentParser(description="Phase 3 — Run OCR pipeline and model comparison")
    parser.add_argument("--use-corpus-sample", action="store_true",
                        help="Sample additional pages from the full corpus instead of using "
                             "the existing 40-page Phase 1 ground truth")
    parser.add_argument("--sample-size", type=int, default=100,
                        help="Number of pages to sample if --use-corpus-sample is set")
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    parser.add_argument("--ground-truth-dir", type=Path, default=Path("data/ground_truth"))
    parser.add_argument("--sample-dir", type=Path, default=Path("data/ocr_sample"))
    parser.add_argument("--preprocessed-dir", type=Path, default=None,
                        help="Defaults to outputs/phase2/preprocessed_images (ground truth) "
                             "or outputs/phase2/preprocessed_sample (corpus sample)")
    parser.add_argument("--models", type=str, default="claude,tesseract",
                        help="Comma-separated OCR models to run (tesseract,claude,gemini,qwen3vl,easyocr)")
    parser.add_argument("--vision-model", type=str, default="qwen3-vl:8b")
    parser.add_argument("--api-key-env", type=str, default=None,
                        help="Env var name for the API key, overriding the chosen backend's "
                             "standard one (e.g. ANTHROPIC_API_KEY, GOOGLE_API_KEY)")
    parser.add_argument("--out-root", type=Path, default=Path("outputs/phase3"))
    parser.add_argument("--force", action="store_true",
                        help="Re-run every step even if its output already exists")
    args = parser.parse_args()

    python = sys.executable
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    # ── Step 0: (Optional) sample additional pages from the full corpus ────────
    raw_dir = args.ground_truth_dir
    if args.use_corpus_sample:
        manifest = args.sample_dir / "manifest.csv"
        manifest_count = None
        if manifest.exists():
            import pandas as pd
            manifest_count = len(pd.read_csv(manifest))
        counts_match = manifest_count == args.sample_size

        if counts_match and not args.force:
            print(f"\n[SKIP] Step 0 — {manifest} already has {manifest_count} page(s) "
                  f"matching the requested --sample-size. Use --force to resample.")
        else:
            if manifest.exists() and manifest_count != args.sample_size:
                print(f"[NOTICE] {manifest} has {manifest_count} page(s), but --sample-size is "
                      f"{args.sample_size} — resampling (0_sample_corpus_for_ocr.py also clears "
                      f"stale leftover files from any previous larger sample).")
            run_step(
                "Step 0 — Sample pages from the full corpus",
                [python, str(SCRIPT_DIR / "0_sample_corpus_for_ocr.py"),
                 "--corpus-dir", str(args.corpus_dir), "--n", str(args.sample_size),
                 "--out", str(args.sample_dir),
                 "--exclude-ground-truth", str(args.ground_truth_dir / "manifest.csv")],
            )
        raw_dir = args.sample_dir

    if not raw_dir.exists():
        sys.exit(
            f"ERROR: {raw_dir} not found.\n"
            "Run Phase 1 first (python phase1/run_phase1.py), or pass --use-corpus-sample."
        )

    # ── Step 1: Preprocess images (pulled in from phase2) ──────────────────────
    preprocessed_dir = args.preprocessed_dir or (
        Path("outputs/phase2/preprocessed_sample") if args.use_corpus_sample
        else Path("outputs/phase2/preprocessed_images")
    )
    expected_count = len(list(raw_dir.glob("*.jpg")))
    actual_count = len(list(preprocessed_dir.glob("*.png"))) if preprocessed_dir.exists() else 0
    counts_match = preprocessed_dir.exists() and actual_count == expected_count and expected_count > 0

    if counts_match and not args.force:
        print(f"\n[SKIP] Step 1 — {preprocessed_dir} already has {actual_count} preprocessed "
              f"page(s) matching the current {expected_count}-page sample. Use --force to redo.")
    else:
        if preprocessed_dir.exists():
            if actual_count != expected_count and actual_count > 0:
                print(f"[NOTICE] {preprocessed_dir} has {actual_count} stale page(s), but the "
                      f"current sample is {expected_count} page(s) — clearing and regenerating "
                      f"to avoid silently processing/paying for the wrong page count.")
            # CRITICAL: clear stale leftovers before regenerating. Without this,
            # re-running with a SMALLER --sample-size than a previous run left
            # the old, larger set of preprocessed images in place — 1_run_ocr.py
            # globs *.png blindly, so it would OCR every leftover file too,
            # silently processing (and paying for) far more pages than requested.
            import shutil
            shutil.rmtree(preprocessed_dir)
        run_step(
            "Step 1 — Preprocess images (phase2)",
            [python, str(PHASE2_DIR / "1_preprocess_images.py"),
             "--input-dir", str(raw_dir), "--out-dir", str(preprocessed_dir)],
        )

    # ── Step 2: Run OCR for each requested model ────────────────────────────────
    model_to_backend = {"qwen3vl": "ollama", "gemini": "gemini", "claude": "anthropic"}
    for model in models:
        backend = model_to_backend.get(model)
        if backend:
            try:
                check_backend_ready(backend, api_key_env=args.api_key_env)
            except SystemExit as e:
                sys.exit(
                    f"\n{e}\n\n"
                    f"Requested --models includes '{model}', which needs the '{backend}' backend.\n"
                    f"Either fix it and re-run, or remove '{model}' from --models."
                )

    for model in models:
        log_path = args.out_root / f"{model}_ocr_log.csv"
        if log_path.exists() and not args.force:
            print(f"\n[SKIP] Step 2 ({model}) — {log_path} already exists. Use --force to redo.")
            continue
        cmd = [python, str(SCRIPT_DIR / "1_run_ocr.py"),
               "--input-dir", str(preprocessed_dir), "--model", model, "--out-root", str(args.out_root)]
        if model == "qwen3vl":
            cmd += ["--vision-model", args.vision_model]
        if model in ("gemini", "claude") and args.api_key_env:
            cmd += ["--api-key-env", args.api_key_env]
        run_step(f"Step 2 — Run OCR ({model})", cmd)

    # ── Step 3: Compare all models against reference text ──────────────────────
    metrics_dir = args.out_root / "metrics"
    summary_path = metrics_dir / "ocr_summary.csv"
    if summary_path.exists() and not args.force:
        print(f"\n[SKIP] Step 3 — {summary_path} already exists. Use --force to redo.")
    else:
        run_step(
            "Step 3 — Compare OCR models against reference text",
            [python, str(SCRIPT_DIR / "2_compare_ocr_models.py"),
             "--reference-dir", str(raw_dir), "--ocr-root", str(args.out_root),
             "--metrics-dir", str(metrics_dir)],
        )

    # ── Done ─────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  Phase 3 complete.")
    print(f"{'=' * 70}")
    print(f"""
Results saved in: {args.out_root}/
Summary: {summary_path}

Next step (Phase 4 — point --ocr-output at the real OCR output above instead of synthetic data):
  python phase4/run_phase4.py --ocr-output-a {args.out_root}/{models[0]} --model-name-a {models[0]} --skip-synthetic
""")


if __name__ == "__main__":
    main()
