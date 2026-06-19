"""
phase1/1_build_profile.py
━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 1, Step 1 — Build corpus_profile.json

Scans every book folder in the downloaded corpus, reads its meta.json,
and compiles a single corpus_profile.json index with per-book page
counts and quality labels. This is the input file that
2_corpus_characterize.py and 3_sample_ground_truth.py both rely on.

Usage
-----
  python phase1/1_build_profile.py --corpus-dir data/corpus/

Output
------
  data/corpus/corpus_profile.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

QUALITY_LABELS = {
    0: "no_text",
    1: "not_proofread",
    2: "problematic",
    3: "human_proofread",
    4: "validated",
}


def build_profile(corpus_dir: Path) -> list:
    records = []

    for book_dir in sorted(corpus_dir.iterdir()):
        if not book_dir.is_dir() or book_dir.name.startswith("."):
            continue

        meta_path = book_dir / "meta.json"
        if not meta_path.exists():
            continue

        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"  Skipping {book_dir.name}: {e}")
            continue

        pages = meta.get("pages", {})
        qualities = [p.get("quality", -1) for p in pages.values()]
        avg_quality = sum(qualities) / len(qualities) if qualities else 0
        rounded = round(avg_quality)

        records.append({
            "index_title": book_dir.name,
            "page_count": meta.get("total_pages", 0),
            "saved": meta.get("saved", 0),
            "quality_avg": round(avg_quality, 2),
            "quality_label": QUALITY_LABELS.get(rounded, "unknown"),
            "telugu_ratio": None,  # not provided by source metadata
            "decision": "approved",
        })

    return records


def main():
    parser = argparse.ArgumentParser(description="Phase 1 — Build corpus_profile.json")
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"),
                        help="Root directory containing downloaded book folders")
    args = parser.parse_args()

    if not args.corpus_dir.exists():
        raise SystemExit(
            f"{args.corpus_dir} not found. Run 0_download_corpus.py first."
        )

    print(f"Scanning {args.corpus_dir}/ for book folders...")
    records = build_profile(args.corpus_dir)

    out_path = args.corpus_dir / "corpus_profile.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    labels = Counter(r["quality_label"] for r in records)
    total_pages = sum(r["page_count"] for r in records)

    print(f"\nBuilt profile for {len(records)} books ({total_pages:,} total pages).")
    print("Quality distribution:", dict(labels))
    print(f"\nSaved: {out_path}")
    print("\nNext step:")
    print(f"  python phase1/2_corpus_characterize.py --profile-json {out_path}")


if __name__ == "__main__":
    main()
