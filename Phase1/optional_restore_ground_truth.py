"""
phase1/optional_restore_ground_truth.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPTIONAL — Restore Ground Truth Files to Match a Completed Annotation

Use this when data/ground_truth/ has been overwritten with a DIFFERENT
random 40-page sample (e.g. from running 3_sample_ground_truth.py or
run_phase1.py with --force), but you still have your old, already
completed annotation_template.csv listing the exact book+page
combinations you originally reviewed.

This re-extracts those EXACT pages from the full corpus and rebuilds
data/ground_truth/ to match your old annotation work — restoring the
correct .jpg/.txt pairs so your completed annotations are usable again.

It does NOT regenerate or touch the annotation_template.csv content
itself (your completed Confirmed/corrections/notes are preserved
exactly as they are in the file you point it at) — it only restores
the corpus to match the manifest.

Usage
-----
  python phase1/optional_restore_ground_truth.py \\
      --old-annotation-csv path/to/your/backed_up/annotation_template.csv \\
      --corpus-dir data/corpus/ \\
      --out data/ground_truth/

If your old completed CSV isn't already inside data/ground_truth/ (e.g.
you saved a backup copy elsewhere after the --force wiped it), point
--old-annotation-csv at wherever that backup actually lives.

Output
------
  data/ground_truth/
    <filename>.jpg          Restored scan images (one per locked page)
    <filename>.txt          Restored Wikisource transcriptions
    annotation_template.csv Your original completed annotation, copied in
    manifest.csv            Regenerated to match (book/page/quality columns)
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Restore ground truth files to match a completed annotation CSV")
    parser.add_argument("--old-annotation-csv", type=Path, required=True,
                        help="Path to your previously completed annotation_template.csv")
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"),
                        help="Full corpus directory to re-extract pages from")
    parser.add_argument("--out", type=Path, default=Path("data/ground_truth"),
                        help="Destination directory (will be populated to match the old annotation)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing files in --out without asking")
    args = parser.parse_args()

    if not args.old_annotation_csv.exists():
        raise SystemExit(f"{args.old_annotation_csv} not found.")

    if not args.corpus_dir.exists():
        raise SystemExit(f"{args.corpus_dir} not found. Download the corpus first.")

    df = pd.read_csv(args.old_annotation_csv, encoding="utf-8-sig")
    required_cols = {"book", "page", "filename"}
    if not required_cols.issubset(df.columns):
        raise SystemExit(
            f"{args.old_annotation_csv} is missing expected columns {required_cols}. "
            "Is this really an annotation_template.csv?"
        )

    print(f"Restoring {len(df)} pages listed in: {args.old_annotation_csv}\n")

    args.out.mkdir(parents=True, exist_ok=True)

    # Warn if the destination already has DIFFERENT files that would be overwritten
    existing_stems = {p.stem for p in args.out.glob("*.jpg")} | {p.stem for p in args.out.glob("*.txt")}
    locked_stems = set(df["filename"].astype(str))
    conflicting = existing_stems - locked_stems
    if conflicting and not args.force:
        print(f"⚠️  {args.out} currently contains {len(conflicting)} file(s) NOT in the old annotation "
              "(likely the accidental resample):")
        for c in sorted(conflicting)[:10]:
            print(f"   - {c}")
        if len(conflicting) > 10:
            print(f"   ... and {len(conflicting) - 10} more")
        confirm = input(
            f"\nProceed and restore the correct {len(df)} pages? The {len(conflicting)} "
            f"unrelated file(s) above will be left in place unless you clean them up "
            f"separately. [y/N]: "
        ).strip().lower()
        if confirm != "y":
            raise SystemExit("Aborted.")

    restored = 0
    missing_from_corpus = []

    for _, row in df.iterrows():
        book = str(row["book"])
        page = int(row["page"])
        stem = str(row["filename"])

        src_jpg = args.corpus_dir / book / f"page_{page:04d}.jpg"
        src_txt = args.corpus_dir / book / f"page_{page:04d}.txt"
        dst_jpg = args.out / f"{stem}.jpg"
        dst_txt = args.out / f"{stem}.txt"

        if not src_jpg.exists() or not src_txt.exists():
            missing_from_corpus.append(stem)
            continue

        shutil.copy2(src_jpg, dst_jpg)
        shutil.copy2(src_txt, dst_txt)
        restored += 1

    print(f"Restored {restored}/{len(df)} pages from the corpus.")

    if missing_from_corpus:
        print(f"\n⚠️  {len(missing_from_corpus)} page(s) could not be found in {args.corpus_dir} "
              "(book folder or page file missing):")
        for m in missing_from_corpus:
            print(f"   - {m}")

    # Copy the completed annotation CSV itself into the restored folder
    dst_annotation = args.out / "annotation_template.csv"
    if dst_annotation.resolve() != args.old_annotation_csv.resolve():
        shutil.copy2(args.old_annotation_csv, dst_annotation)
        print(f"\nCopied your completed annotation to: {dst_annotation}")

    # Regenerate manifest.csv to match, for any downstream script that reads it
    # (e.g. phase3/0_sample_corpus_for_ocr.py's --exclude-ground-truth)
    manifest_cols = [c for c in ["book", "page", "quality", "quality_label"] if c in df.columns]
    if manifest_cols:
        manifest = df[manifest_cols].copy()
        manifest_path = args.out / "manifest.csv"
        manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
        print(f"Regenerated: {manifest_path}")

    print(f"\n{'=' * 60}")
    if restored == len(df):
        print("RESTORE COMPLETE: all pages matched and restored successfully.")
    else:
        print(f"RESTORE PARTIAL: {restored}/{len(df)} pages restored — see missing list above.")
    print(f"{'=' * 60}")
    print("\nRecommended next step — verify the restore matches exactly:")
    print("  python phase1/optional_lock_ground_truth.py")


if __name__ == "__main__":
    main()
