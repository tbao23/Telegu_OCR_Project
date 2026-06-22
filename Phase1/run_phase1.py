"""
phase1/run_phase1.py
━━━━━━━━━━━━━━━━━━━━
Phase 1 Orchestrator — Runs All Steps in Order

Runs the full Phase 1 pipeline end to end:
  0. Download the corpus from HuggingFace
  1. Build corpus_profile.json from meta.json files
  2. Generate corpus statistics and charts
  3. Sample 40 pages for manual ground-truth annotation

This script does NOT render the .qmd report and does NOT perform manual
annotation — those two steps still require a human (rendering needs Quarto
installed; annotation needs a person to read Telugu and compare it to
each scan). Everything else is automated here.

EVERY step also skips itself automatically if its output already exists
on disk — re-running this script will NOT overwrite an already-completed
annotation_template.csv (Step 3) or redo Steps 1-2 unnecessarily. Pass
--force to override and redo every step regardless (this WILL wipe any
completed annotation work in Step 3 — back it up first if you have one).

Usage
-----
  # Default — prompts interactively whether to download the corpus:
  python phase1/run_phase1.py

  # Token via environment variable (recommended for repeatable runs):
  set HF_TOKEN=hf_xxxxxxxxxxxx          (Windows PowerShell: $env:HF_TOKEN="hf_xxx")
  python phase1/run_phase1.py

  # Force-skip the download step without being asked (corpus already on disk):
  python phase1/run_phase1.py --skip-download

  # Force a (re)download without being asked:
  python phase1/run_phase1.py --download

  # Change the ground-truth sample size:
  python phase1/run_phase1.py --skip-download --n 50

Output
------
  data/corpus/                  Downloaded corpus + corpus_profile.json
  outputs/                      corpus_stats.json + 3 charts
  data/ground_truth/            40 sampled pages + annotation_template.csv
"""

import argparse
import getpass
import os
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
    parser = argparse.ArgumentParser(description="Phase 1 — Run all steps in order")
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"),
                        help="Corpus directory (default: data/corpus)")
    parser.add_argument("--ground-truth-dir", type=Path, default=Path("data/ground_truth"),
                        help="Ground truth output directory (default: data/ground_truth)")
    parser.add_argument("--n", type=int, default=40,
                        help="Number of ground-truth pages to sample (default: 40)")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token. If omitted, checks HF_TOKEN env var, then prompts.")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip step 0 without prompting — use if the corpus is already downloaded")
    parser.add_argument("--download", action="store_true",
                        help="Run step 0 without prompting — forces a (re)download")
    parser.add_argument("--force", action="store_true",
                        help="Re-run every step even if its output already exists "
                             "(WARNING: this will overwrite an already-annotated "
                             "annotation_template.csv in Step 3)")
    parser.add_argument("--skip-ground-truth-sample", action="store_true",
                        help="Skip Step 3 (random ground-truth sampling) entirely — use this when "
                             "you're going to restore a previous annotation instead (see "
                             "optional_restore_ground_truth.py), so you never waste a sample that's "
                             "about to be overwritten anyway")
    parser.add_argument("--skip-image-stats", action="store_true",
                        help="Skip real image resolution/DPI sampling in Step 2 (faster, but the "
                             "assignment requires 300 DPI minimum input — without this check you "
                             "won't know if any pages need upscaling before Phase 2/3. Only skip "
                             "this for quick iteration; do a real run without it before submission.")
    args = parser.parse_args()

    python = sys.executable

    # ── Step 0: Download (with interactive prompt if not explicitly set) ───────
    do_download = None
    if args.skip_download and args.download:
        sys.exit("--skip-download and --download cannot both be set.")
    elif args.skip_download:
        do_download = False
    elif args.download:
        do_download = True
    else:
        # No flag given — ask interactively
        corpus_already_exists = args.corpus_dir.exists() and any(args.corpus_dir.iterdir())
        default_hint = "n" if corpus_already_exists else "y"
        if corpus_already_exists:
            print(f"\nFound existing files in {args.corpus_dir} — it looks like the corpus may already be downloaded.")
        answer = input(
            f"Download the corpus now? [y/n] (default: {default_hint}): "
        ).strip().lower()
        if not answer:
            answer = default_hint
        do_download = answer.startswith("y")

    if not do_download:
        print("\nSkipping download step. Using existing corpus at", args.corpus_dir)
        if not args.corpus_dir.exists():
            sys.exit(f"{args.corpus_dir} does not exist. Re-run and choose 'y' to download it.")
    else:
        token = args.token or os.environ.get("HF_TOKEN")
        if not token:
            print("\nNo --token given and HF_TOKEN environment variable not set.")
            print("Get a free token at: https://huggingface.co/settings/tokens\n")
            token = getpass.getpass("Paste your HuggingFace token (input hidden): ").strip()
        if not token:
            sys.exit("No token provided. Aborting.")

        run_step(
            "Step 0 — Download Telugu OCR Corpus",
            [python, str(SCRIPT_DIR / "0_download_corpus.py"),
             "--out", str(args.corpus_dir), "--token", token],
        )

    # ── Step 1: Build profile ───────────────────────────────────────────────
    profile_json = args.corpus_dir / "corpus_profile.json"
    if profile_json.exists() and not args.force:
        print(f"\n[SKIP] Step 1 — {profile_json} already exists. Use --force to redo.")
    else:
        run_step(
            "Step 1 — Build corpus_profile.json",
            [python, str(SCRIPT_DIR / "1_build_profile.py"),
             "--corpus-dir", str(args.corpus_dir)],
        )

    # ── Step 2: Characterize ────────────────────────────────────────────────
    corpus_stats = Path("outputs/phase1/corpus_stats.json")
    if corpus_stats.exists() and not args.force:
        print(f"\n[SKIP] Step 2 — {corpus_stats} already exists. Use --force to redo.")
    else:
        cmd2 = [python, str(SCRIPT_DIR / "2_corpus_characterize.py"),
                "--profile-json", str(profile_json)]
        if args.skip_image_stats:
            cmd2.append("--skip-images")
        run_step("Step 2 — Corpus Characterization (stats + charts)", cmd2)

    # ── Step 3: Sample ground truth ─────────────────────────────────────────
    annotation_csv = args.ground_truth_dir / "annotation_template.csv"
    if args.skip_ground_truth_sample:
        print("\n[SKIP] Step 3 — skipped via --skip-ground-truth-sample "
              "(restoring a previous annotation instead).")
    elif annotation_csv.exists() and not args.force:
        print(f"\n[SKIP] Step 3 — {annotation_csv} already exists. Use --force to redo.")
        print("       (--force will OVERWRITE this file, wiping any completed annotation work.")
        print("        Back it up first if you have annotation progress saved in it.)")
    else:
        run_step(
            "Step 3 — Sample Ground Truth Pages",
            [python, str(SCRIPT_DIR / "3_sample_ground_truth.py"),
             "--corpus-dir", str(args.corpus_dir),
             "--n", str(args.n),
             "--out", str(args.ground_truth_dir)],
        )

    # ── Done ─────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  Phase 1 automated steps complete.")
    print(f"{'=' * 70}")
    print(f"""
Remaining manual steps:
  1. Open {args.ground_truth_dir / 'annotation_template.csv'} and annotate
     each of the {args.n} sampled pages (compare image to transcription).
  2. Review/update phase1/phase1_report.qmd with your annotation findings.
  3. Render the report:
       quarto render phase1/phase1_report.qmd
""")


if __name__ == "__main__":
    main()
