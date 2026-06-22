"""
phase1/2_corpus_characterize.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 1, Step 2 — Corpus Characterization

Profiles the Telugu OCR corpus (AlbertoChestnut/telugu-ocr or the
professor-provided corpus if it follows the same layout) and produces
quantitative statistics required for the Phase 1 written report.

Usage
-----
  # Download the HuggingFace dataset first, then characterize:
  python phase1/2_corpus_characterize.py --corpus-dir /path/to/dataset/

  # Or let the script download it for you:
  python phase1/2_corpus_characterize.py --download --corpus-dir ./data/corpus/

  # Just characterize from corpus_profile.json (faster, no image I/O):
  python phase1/2_corpus_characterize.py --profile-json corpus_profile.json

Output
------
  outputs/phase1/corpus_stats.json    — Machine-readable summary
  outputs/phase1/quality_dist.png     — Quality level distribution chart
  outputs/phase1/telugu_ratio_dist.png — Telugu script ratio histogram
  outputs/phase1/size_dist.png        — Pages-per-book distribution
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ── Quality level labels (Wikisource ProofreadPage) ──────────────────────────
QUALITY_LABELS = {
    0: "no_text",
    1: "not_proofread",
    2: "problematic",
    3: "human_proofread",
    4: "validated",
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset download
# ─────────────────────────────────────────────────────────────────────────────

def download_dataset(corpus_dir: Path) -> None:
    """Download AlbertoChestnut/telugu-ocr from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("Install huggingface_hub: pip install huggingface_hub")

    print(f"Downloading dataset to {corpus_dir} (~11 GB, may take a while)…")
    snapshot_download(
        repo_id="AlbertoChestnut/telugu-ocr",
        repo_type="dataset",
        local_dir=str(corpus_dir),
    )
    print("Download complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Profile-based statistics (fast path — no image I/O)
# ─────────────────────────────────────────────────────────────────────────────

def load_corpus_profile(profile_path: Path) -> pd.DataFrame:
    """Load corpus_profile.json and return a DataFrame of approved books."""
    with open(profile_path, encoding="utf-8") as f:
        records = json.load(f)

    # Support both list-of-dicts and dict-of-dicts formats
    if isinstance(records, dict):
        records = list(records.values())

    df = pd.DataFrame(records)
    return df


def summarize_from_profile(df: pd.DataFrame) -> dict:
    """Compute corpus-level statistics from the profile DataFrame."""
    approved = df[df["decision"] == "approved"].copy() if "decision" in df.columns else df.copy()
    excluded = df[df["decision"] == "excluded"].copy() if "decision" in df.columns else pd.DataFrame()

    # Page count totals
    total_pages_approved = int(approved["page_count"].sum()) if "page_count" in approved.columns else 0

    # Quality distribution across approved books
    quality_counts = (
        Counter(approved["quality_label"].dropna().tolist())
        if "quality_label" in approved.columns else {}
    )

    # Telugu ratio statistics
    telugu_ratios = (
        approved["telugu_ratio"].dropna()
        if "telugu_ratio" in approved.columns else pd.Series([], dtype=float)
    )

    # Exclusion reasons
    excl_reasons = (
        Counter(excluded["exclusion_reason"].dropna().tolist())
        if "exclusion_reason" in excluded.columns and len(excluded) > 0 else {}
    )

    stats = {
        "total_books_in_manifest": len(df),
        "approved_books": len(approved),
        "excluded_books": len(excluded),
        "total_pages_approved": total_pages_approved,
        "quality_distribution": dict(quality_counts),
        "telugu_ratio": {
            "mean": round(float(telugu_ratios.mean()), 4) if len(telugu_ratios) > 0 else "N/A",
            "median": round(float(telugu_ratios.median()), 4) if len(telugu_ratios) > 0 else "N/A",
            "min": round(float(telugu_ratios.min()), 4) if len(telugu_ratios) > 0 else "N/A",
            "max": round(float(telugu_ratios.max()), 4) if len(telugu_ratios) > 0 else "N/A",
            "pct_above_90": round(float((telugu_ratios > 0.9).mean() * 100), 1) if len(telugu_ratios) > 0 else "N/A",
        },
        "pages_per_book": {
            "mean": round(float(approved["page_count"].mean()), 1) if "page_count" in approved.columns else "N/A",
            "median": float(approved["page_count"].median()) if "page_count" in approved.columns else "N/A",
            "min": int(approved["page_count"].min()) if "page_count" in approved.columns else "N/A",
            "max": int(approved["page_count"].max()) if "page_count" in approved.columns else "N/A",
        },
        "exclusion_reasons": dict(excl_reasons),
    }
    return stats, approved


# ─────────────────────────────────────────────────────────────────────────────
# Image-level statistics (slower — reads actual JPEG files)
# ─────────────────────────────────────────────────────────────────────────────

def sample_image_stats(corpus_dir: Path, n_books: int = 20, pages_per_book: int = 3) -> dict:
    """
    Sample a subset of images to estimate resolution distribution.
    Reads n_books * pages_per_book images; returns DPI/size statistics.
    """
    book_dirs = sorted([d for d in corpus_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    if not book_dirs:
        return {}
    # Random sample of books
    rng = np.random.default_rng(42)
    sampled_books = rng.choice(book_dirs, size=min(n_books, len(book_dirs)), replace=False)

    widths, heights, file_sizes_kb = [], [], []

    for book_dir in tqdm(sampled_books, desc="Sampling images"):
        jpegs = sorted(book_dir.glob("*.jpg"))[:pages_per_book]
        for img_path in jpegs:
            try:
                with Image.open(img_path) as img:
                    widths.append(img.width)
                    heights.append(img.height)
                file_sizes_kb.append(img_path.stat().st_size / 1024)
            except Exception:
                continue

    if not widths:
        return {}

    return {
        "sample_size": len(widths),
        "width_px": {
            "mean": round(np.mean(widths)),
            "median": round(np.median(widths)),
            "min": int(np.min(widths)),
            "max": int(np.max(widths)),
        },
        "height_px": {
            "mean": round(np.mean(heights)),
            "median": round(np.median(heights)),
            "min": int(np.min(heights)),
            "max": int(np.max(heights)),
        },
        "file_size_kb": {
            "mean": round(float(np.mean(file_sizes_kb)), 1),
            "median": round(float(np.median(file_sizes_kb)), 1),
        },
    }


def estimate_character_count(corpus_dir: Path, n_sample: int = 500) -> dict:
    """
    Sample .txt files to estimate average characters per page and
    project total character count across the corpus.
    """
    all_txts = list(corpus_dir.glob("*/*.txt"))
    if not all_txts:
        return {}
    rng = np.random.default_rng(42)
    sampled = rng.choice(all_txts, size=min(n_sample, len(all_txts)), replace=False)

    char_counts = []
    for txt_path in tqdm(sampled, desc="Sampling .txt files"):
        try:
            text = Path(txt_path).read_text(encoding="utf-8", errors="ignore")
            char_counts.append(len(text))
        except Exception:
            continue

    if not char_counts:
        return {}

    avg_chars = np.mean(char_counts)
    # Total page pairs in the dataset
    total_pairs = len(all_txts)

    return {
        "sample_size": len(char_counts),
        "avg_chars_per_page": round(float(avg_chars)),
        "median_chars_per_page": round(float(np.median(char_counts))),
        "estimated_total_chars": round(float(avg_chars * total_pairs)),
        "estimated_total_pages": total_pairs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────────────────────────────────────

def plot_quality_distribution(stats: dict, out_dir: Path) -> None:
    label_order = ["no_text", "not_proofread", "problematic", "human_proofread", "validated"]
    q = stats.get("quality_distribution", {})
    counts = [q.get(lbl, 0) for lbl in label_order]
    colors = ["#d9534f", "#f0ad4e", "#e67e22", "#5cb85c", "#2980b9"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(label_order, counts, color=colors, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_xlabel("Wikisource ProofreadPage Quality Level", fontsize=11)
    ax.set_ylabel("Number of Books", fontsize=11)
    ax.set_title("Quality Distribution of Approved Telugu OCR Books", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    path = out_dir / "quality_dist.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_telugu_ratio_histogram(approved_df: pd.DataFrame, out_dir: Path) -> None:
    ratios = approved_df["telugu_ratio"].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratios, bins=20, color="#2980b9", edgecolor="white", linewidth=0.6)
    ax.axvline(ratios.mean(), color="#e74c3c", linestyle="--", label=f"Mean = {ratios.mean():.2f}")
    ax.axvline(0.80, color="#27ae60", linestyle=":", label="Inclusion threshold (0.80)")
    ax.set_xlabel("Telugu Script Ratio", fontsize=11)
    ax.set_ylabel("Number of Books", fontsize=11)
    ax.set_title("Distribution of Telugu Character Ratio Across Approved Books", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = out_dir / "telugu_ratio_dist.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_pages_per_book(approved_df: pd.DataFrame, out_dir: Path) -> None:
    pages = approved_df["page_count"].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pages, bins=30, color="#8e44ad", edgecolor="white", linewidth=0.6)
    ax.axvline(pages.median(), color="#e74c3c", linestyle="--",
               label=f"Median = {pages.median():.0f} pages")
    ax.set_xlabel("Pages per Book", fontsize=11)
    ax.set_ylabel("Number of Books", fontsize=11)
    ax.set_title("Distribution of Book Length (Scan Pages)", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = out_dir / "pages_per_book_dist.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 1 — Telugu Corpus Characterization")
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"),
                        help="Root directory of the downloaded dataset")
    parser.add_argument("--profile-json", type=Path, default=None,
                        help="Path to corpus_profile.json (overrides --corpus-dir lookup)")
    parser.add_argument("--download", action="store_true",
                        help="Download dataset from HuggingFace before characterizing")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/phase1"),
                        help="Directory for output files")
    parser.add_argument("--skip-images", action="store_true",
                        help="Skip image sampling (faster, no PIL required)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Download (optional) ──────────────────────────────────────────────────
    if args.download:
        download_dataset(args.corpus_dir)

    # ── Load profile ─────────────────────────────────────────────────────────
    profile_path = args.profile_json or (args.corpus_dir / "corpus_profile.json")
    if not profile_path.exists():
        sys.exit(f"corpus_profile.json not found at {profile_path}. "
                 "Run with --download or pass --profile-json.")

    print(f"\nLoading profile: {profile_path}")
    df = load_corpus_profile(profile_path)
    stats, approved_df = summarize_from_profile(df)

    # ── Image sampling ────────────────────────────────────────────────────────
    if not args.skip_images and args.corpus_dir.exists():
        print("\nSampling images for resolution statistics…")
        img_stats = sample_image_stats(args.corpus_dir)
        stats["image_resolution"] = img_stats

        print("\nSampling .txt files for character count estimate…")
        char_stats = estimate_character_count(args.corpus_dir)
        stats["character_count_estimate"] = char_stats
    else:
        print("\nSkipping image/text sampling (--skip-images or dataset/ not found).")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CORPUS SUMMARY")
    print("=" * 60)
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_json = args.out_dir / "corpus_stats.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nSaved statistics: {out_json}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots…")
    plot_quality_distribution(stats, args.out_dir)
    if "telugu_ratio" in approved_df.columns:
        plot_telugu_ratio_histogram(approved_df, args.out_dir)
    if "page_count" in approved_df.columns:
        plot_pages_per_book(approved_df, args.out_dir)

    print("\nPhase 1 characterization complete.")
    print("\nNext step:")
    print(f"  python phase1/3_sample_ground_truth.py --corpus-dir {args.corpus_dir}/")


if __name__ == "__main__":
    main()
