# Phase 1 – Corpus Characterization and Problem Scoping

## Objective

Understand the corpus before writing any OCR code: profile its scale, quality
distribution, and script-specific challenges, and produce a manually
annotated ground-truth sample for later evaluation.

## Pipeline Steps

### 0. Download the corpus

```
python phase1/0_download_corpus.py
```

Downloads the Telugu Wikisource–derived corpus from HuggingFace
(`AlbertoChestnut/telugu-ocr`). Requires a free HuggingFace account/token —
you'll be prompted. ~10.3 GB, handles rate limits automatically.

### 1. Build the corpus profile index

```
python phase1/1_build_profile.py --corpus-dir data/corpus/
```

Scans every book's `meta.json` and compiles `corpus_profile.json` — the
index that Steps 2 and 3 both read from.

### 2. Characterize the corpus

```
python phase1/2_corpus_characterize.py --profile-json data/corpus/corpus_profile.json
```

Generates corpus-wide statistics and charts (quality distribution, pages
per book, etc.).

### 3. Sample ground truth pages

```
python phase1/3_sample_ground_truth.py --corpus-dir data/corpus/ --n 40 --out data/ground_truth/
```

Selects a stratified sample across quality levels for manual annotation.

### Or run all four steps at once:

```
python phase1/run_phase1.py
```

## Manual Step (not automated)

Open `data/ground_truth/annotation_template.csv` and, for each sampled
page, compare the image against its existing Wikisource transcription.
Mark `Confirmed` if accurate, or supply a correction if not. Note any scan
artifacts observed (fading, blur, etc.) and a 1–5 quality score.

## Inputs

None — this phase starts from the raw HuggingFace dataset.

## Outputs

```
data/corpus/                       Downloaded corpus + corpus_profile.json
outputs/corpus_stats.json          Corpus-wide statistics
outputs/quality_dist.png           Quality label distribution chart
outputs/pages_per_book_dist.png    Book length distribution chart
data/ground_truth/                 40 sampled pages + annotation_template.csv
phase1/phase1_report.qmd           Written deliverable (render with Quarto)
```

## Render the Report

```
quarto render phase1/phase1_report.qmd
```

Produces `phase1_report.html` and `phase1_report.pdf`. Requires Quarto and
TinyTeX (`quarto install tinytex`) — see root README Prerequisites table.

## Dependencies

pip install huggingface_hub pandas numpy matplotlib tqdm
