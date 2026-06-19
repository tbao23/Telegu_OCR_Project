"""
phase1/0_download_corpus.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 1, Step 0 — Download the Telugu OCR Corpus

Downloads the teammate-assembled corpus (AlbertoChestnut/telugu-ocr) from
HuggingFace Hub. Requires a free HuggingFace account and access token.

Setup
-----
  1. Create a free account at https://huggingface.co/join
  2. Generate a token at https://huggingface.co/settings/tokens
     (no special permissions needed — default read access is fine)
  3. Run this script and paste your token when prompted, OR pass it
     directly with --token

Usage
-----
  python phase1/0_download_corpus.py
  python phase1/0_download_corpus.py --token hf_xxxxxxxxxxxx
  python phase1/0_download_corpus.py --out data/corpus/

Notes
-----
  - The full corpus is ~10.3 GB across ~51,000 files. Expect 1-2 hours
    depending on connection speed.
  - HuggingFace rate-limits free accounts to 5,000 requests per 5-minute
    window. huggingface_hub >= 1.0 handles this automatically — it will
    print "Rate limited. Waiting Xs..." and resume on its own. Do not
    close the terminal; just let it run.
  - Safe to re-run if interrupted — already-downloaded files are skipped.
  - One book ("Cheppulu_Kudutu_Kudutu...") is excluded by default because
    its folder name exceeds Windows' 260-character path limit. Use
    --include-cheppulu to attempt it anyway (may fail on Windows).
"""

import argparse
import getpass
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Phase 1 — Download Telugu OCR Corpus")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace access token (hf_...). If omitted, you'll be prompted.")
    parser.add_argument("--out", type=Path, default=Path("data/corpus"),
                        help="Destination directory for the downloaded corpus")
    parser.add_argument("--repo-id", type=str, default="AlbertoChestnut/telugu-ocr",
                        help="HuggingFace dataset repo ID")
    parser.add_argument("--include-cheppulu", action="store_true",
                        help="Attempt to download the long-path book (may fail on Windows)")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub not installed. Run: pip install --upgrade huggingface_hub")

    token = args.token
    if not token:
        print("A HuggingFace token is required to avoid aggressive rate limiting.")
        print("Get one free at: https://huggingface.co/settings/tokens\n")
        token = getpass.getpass("Paste your HuggingFace token (input hidden): ").strip()

    if not token:
        sys.exit("No token provided. Aborting.")

    ignore_patterns = None if args.include_cheppulu else ["*Cheppulu*"]

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {args.repo_id} to {args.out}/ ...")
    print("This is ~10.3 GB and may take 1-2 hours. Rate-limit pauses are normal and automatic.\n")

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(args.out),
        token=token,
        ignore_patterns=ignore_patterns,
    )

    print("\nDownload complete.")
    print(f"Corpus available at: {args.out}/")
    print("\nNext step:")
    print(f"  python phase1/1_build_profile.py --corpus-dir {args.out}/")


if __name__ == "__main__":
    main()
