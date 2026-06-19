"""
phase1/3_sample_ground_truth.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 1, Step 3 — Stratified Ground Truth Sample Selection

Selects 30–50 pages stratified across the quality spectrum for manual
annotation. High-quality (level 3–4) pages form the primary evaluation
set; lower-quality pages stress-test the pipeline.

Strategy
--------
  - Quality 4 (validated):       ~40% of sample  → gold standard
  - Quality 3 (human_proofread): ~30% of sample  → reliable ground truth
  - Quality 1–2 (lower quality): ~20% of sample  → error-prone cases
  - Quality 0 (no_text):         ~10% of sample  → blank/image-only edge cases

Usage
-----
  python phase1/3_sample_ground_truth.py \\
      --corpus-dir data/corpus/ \\
      --n 40 \\
      --out data/ground_truth/

Output
------
  data/ground_truth/
    manifest.csv              — List of sampled pages with metadata
    <book>__page_XXXX.jpg     — Copied image files
    <book>__page_XXXX.txt     — Copied reference transcriptions
    annotation_template.csv   — Blank template for manual annotation
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


QUALITY_LABELS = {
    0: "no_text",
    1: "not_proofread",
    2: "problematic",
    3: "human_proofread",
    4: "validated",
}

# Target proportions per quality level
QUALITY_PROPORTIONS = {4: 0.40, 3: 0.30, 2: 0.10, 1: 0.10, 0: 0.10}


def load_all_pages(corpus_dir: Path) -> pd.DataFrame:
    """
    Walk <corpus_dir>/<book>/meta.json files and build a flat DataFrame of
    all available pages with their quality levels.
    """
    if not corpus_dir.exists():
        sys.exit(f"{corpus_dir} not found.")

    rows = []
    for meta_path in tqdm(sorted(corpus_dir.glob("*/meta.json")), desc="Scanning books"):
        book_dir = meta_path.parent
        book_name = book_dir.name
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        pages = meta.get("pages", {})
        for page_num, page_info in pages.items():
            img_path = book_dir / f"page_{int(page_num):04d}.jpg"
            txt_path = book_dir / f"page_{int(page_num):04d}.txt"
            if img_path.exists() and txt_path.exists():
                rows.append(
                    {
                        "book": book_name,
                        "page": int(page_num),
                        "quality": page_info.get("quality", -1),
                        "quality_label": QUALITY_LABELS.get(page_info.get("quality", -1), "unknown"),
                        "img_path": str(img_path),
                        "txt_path": str(txt_path),
                    }
                )

    return pd.DataFrame(rows)


def stratified_sample(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Draw a stratified sample of n pages across quality levels,
    respecting the target proportions defined in QUALITY_PROPORTIONS.
    """
    sampled_parts = []
    total_allocated = 0

    # Sort levels so we allocate deterministically
    for level in sorted(QUALITY_PROPORTIONS.keys(), reverse=True):
        target_count = round(n * QUALITY_PROPORTIONS[level])
        pool = df[df["quality"] == level]
        actual = min(target_count, len(pool))
        if actual > 0:
            sampled_parts.append(pool.sample(n=actual, random_state=int(rng.integers(0, 10000))))
        total_allocated += actual

    result = pd.concat(sampled_parts, ignore_index=True)

    # If we're short due to sparse quality levels, top up from what's available
    if len(result) < n:
        remaining = df[~df.index.isin(result.index)]
        shortfall = n - len(result)
        extra = remaining.sample(n=min(shortfall, len(remaining)),
                                 random_state=int(rng.integers(0, 10000)))
        result = pd.concat([result, extra], ignore_index=True)

    return result.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle


def copy_sample_files(sample_df: pd.DataFrame, out_dir: Path) -> None:
    """Copy sampled image and text files into the output directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Copying files"):
        stem = f"{row['book']}__page_{row['page']:04d}"
        shutil.copy2(row["img_path"], out_dir / f"{stem}.jpg")
        shutil.copy2(row["txt_path"], out_dir / f"{stem}.txt")


def write_annotation_template(sample_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Create a blank annotation CSV for manual review.
    Annotators fill in the 'manual_transcription' and 'notes' columns.
    """
    template = sample_df[["book", "page", "quality", "quality_label"]].copy()
    template["filename"] = (
        template["book"] + "__page_" + template["page"].apply(lambda p: f"{p:04d}")
    )
    template["manual_transcription"] = ""
    template["notes"] = ""          # e.g., "bleed-through", "skewed", "damaged"
    template["scan_quality_score"] = ""  # 1–5 subjective rating
    path = out_dir / "annotation_template.csv"
    template.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Annotation template: {path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 — Ground Truth Sampling")
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"),
                        help="Root directory of the downloaded dataset")
    parser.add_argument("--n", type=int, default=40,
                        help="Number of pages to sample (30–50 recommended)")
    parser.add_argument("--out", type=Path, default=Path("data/ground_truth"),
                        help="Output directory for sampled pages")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"\nScanning corpus at: {args.corpus_dir}")
    pages_df = load_all_pages(args.corpus_dir)
    print(f"Total available page pairs: {len(pages_df):,}")
    print("\nQuality breakdown:")
    print(pages_df.groupby("quality_label").size().to_string())

    print(f"\nSelecting {args.n} pages (stratified by quality)…")
    sample = stratified_sample(pages_df, args.n, rng)
    print("\nSample quality breakdown:")
    print(sample.groupby("quality_label").size().to_string())

    # Save manifest
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.csv"
    sample.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print(f"\nManifest saved: {manifest_path}")

    # Copy files and write annotation template
    copy_sample_files(sample, args.out)
    write_annotation_template(sample, args.out)

    print(f"\nDone. {len(sample)} pages ready for annotation in: {args.out}")
    print("\nNext step:")
    print(f"  Open {args.out / 'annotation_template.csv'} and begin manual annotation.")


if __name__ == "__main__":
    main()
