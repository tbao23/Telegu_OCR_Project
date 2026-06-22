"""
phase3/0_sample_corpus_for_ocr.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 3, Step 0 — Sample Pages from the Full Corpus for OCR Testing

The Phase 1 ground truth sample (40 pages, manually annotated) is too
small to satisfy two separate assignment requirements:
  - Phase 4's LLM validation must scale to a 100+ page sample
  - The final "Processed corpus output" deliverable needs a 500-page
    sample of clean OCR text

This script draws a NEW, larger, stratified-by-quality sample directly
from the full downloaded corpus (data/corpus/) — distinct from the 40
pages already used for manual annotation, so the two samples stay
independent. Since every page in the corpus already carries a
Wikisource transcription, we get usable reference text for CER/WER
"for free" without needing to hand-annotate 100+ pages.

::: Note on ground truth quality :::
These reference texts are Wikisource transcriptions, NOT
manually-reverified ground truth like the Phase 1 sample. They're
reliable for quality levels 3-4 (human_proofread/validated) per the
Phase 1 findings (39/40 confirmed accurate), but should be treated as
provisional for quality levels 1-2. Stratification below defaults to
favoring quality 3-4 for this reason.

Usage
-----
  python phase3/0_sample_corpus_for_ocr.py \\
      --corpus-dir data/corpus/ \\
      --n 100 \\
      --out data/ocr_sample/ \\
      --exclude-ground-truth data/ground_truth/manifest.csv

Output
------
  data/ocr_sample/
    <book>__page_XXXX.jpg   (copied scan image)
    <book>__page_XXXX.txt   (copied Wikisource transcription, used as reference)
    manifest.csv            (sample metadata)
"""

import argparse
import json
import shutil
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

# Favor higher-quality pages since these reference texts aren't manually
# re-verified — different from Phase 1's annotation-focused stratification.
QUALITY_PROPORTIONS = {4: 0.50, 3: 0.35, 2: 0.10, 1: 0.05, 0: 0.0}


def load_all_pages(corpus_dir: Path) -> pd.DataFrame:
    """Scan <corpus_dir>/<book>/meta.json files for all available pages."""
    rows = []
    for meta_path in tqdm(sorted(corpus_dir.glob("*/meta.json")), desc="Scanning books"):
        book_dir = meta_path.parent
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
                # Skip pages with empty transcriptions — no usable reference text
                if not txt_path.read_text(encoding="utf-8", errors="ignore").strip():
                    continue
                rows.append({
                    "book": book_dir.name,
                    "page": int(page_num),
                    "quality": page_info.get("quality", -1),
                    "quality_label": QUALITY_LABELS.get(page_info.get("quality", -1), "unknown"),
                    "img_path": str(img_path),
                    "txt_path": str(txt_path),
                })
    return pd.DataFrame(rows)


def exclude_already_sampled(df: pd.DataFrame, exclude_manifest: Path) -> pd.DataFrame:
    """Remove pages already used in the Phase 1 ground truth sample, so the
    two samples stay independent."""
    if not exclude_manifest or not exclude_manifest.exists():
        return df
    excl = pd.read_csv(exclude_manifest)
    excl_keys = set(zip(excl["book"], excl["page"]))
    mask = ~df.apply(lambda r: (r["book"], r["page"]) in excl_keys, axis=1)
    return df[mask]


def stratified_sample(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    sampled_parts = []
    for level in sorted(QUALITY_PROPORTIONS.keys(), reverse=True):
        target = round(n * QUALITY_PROPORTIONS[level])
        pool = df[df["quality"] == level]
        actual = min(target, len(pool))
        if actual > 0:
            sampled_parts.append(pool.sample(n=actual, random_state=int(rng.integers(0, 10000))))

    result = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else pd.DataFrame()

    if len(result) < n:
        remaining = df[~df.index.isin(result.index)]
        shortfall = n - len(result)
        if len(remaining) > 0:
            extra = remaining.sample(n=min(shortfall, len(remaining)), random_state=int(rng.integers(0, 10000)))
            result = pd.concat([result, extra], ignore_index=True)

    return result.sample(frac=1, random_state=42).reset_index(drop=True)


def copy_sample_files(sample_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Copying files"):
        stem = f"{row['book']}__page_{row['page']:04d}"
        shutil.copy2(row["img_path"], out_dir / f"{stem}.jpg")
        shutil.copy2(row["txt_path"], out_dir / f"{stem}.txt")


def main():
    parser = argparse.ArgumentParser(description="Phase 3 — Sample pages from full corpus for OCR scaling")
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    parser.add_argument("--n", type=int, default=100,
                        help="Number of pages to sample (100 minimum for Phase 4; 500 for final deliverable)")
    parser.add_argument("--out", type=Path, default=Path("data/ocr_sample"))
    parser.add_argument("--exclude-ground-truth", type=Path, default=Path("data/ground_truth/manifest.csv"),
                        help="Phase 1 ground truth manifest, to avoid resampling the same pages")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Scanning corpus at: {args.corpus_dir}")
    pages_df = load_all_pages(args.corpus_dir)
    print(f"Total available page pairs (with non-empty transcription): {len(pages_df):,}")

    pages_df = exclude_already_sampled(pages_df, args.exclude_ground_truth)
    print(f"After excluding Phase 1 ground truth sample: {len(pages_df):,}")

    print(f"\nSelecting {args.n} pages (stratified by quality, favoring 3-4)...")
    sample = stratified_sample(pages_df, args.n, rng)
    print("\nSample quality breakdown:")
    print(sample.groupby("quality_label").size().to_string())

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.csv"
    sample.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print(f"\nManifest saved: {manifest_path}")

    copy_sample_files(sample, args.out)

    print(f"\nDone. {len(sample)} pages ready for OCR testing in: {args.out}")
    print("\nNext step (run Phase 2 preprocessing on this larger sample):")
    print(f"  python phase2/1_preprocess_images.py --input-dir {args.out} --out-dir outputs/phase2/preprocessed_sample")


if __name__ == "__main__":
    main()
