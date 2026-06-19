# Telugu OCR Project — CSCI/DASC 6020: Machine Learning (Summer 2026)

An end-to-end pipeline for converting scanned Telugu document images into clean Unicode text using Large Language Models, with rigorous LLM-assisted validation.

## Project Structure

```
telugu-ocr-project/
├── README.md
├── requirements.txt
├── .gitignore                     # Excludes data/ from version control
├── phase1/                    # Corpus Characterization (Week 1)
│   ├── 0_download_corpus.py       # Download corpus from HuggingFace
│   ├── 1_build_profile.py         # Build corpus_profile.json from meta.json files
│   ├── 2_corpus_characterize.py   # Profile corpus statistics, generate charts
│   ├── 3_sample_ground_truth.py   # Select 30–50 pages for manual annotation
│   └── phase1_report.qmd          # Written deliverable
├── phase2/                    # Preprocessing Pipeline (Week 2)
├── phase3/                    # OCR Pipeline (Week 3)
├── phase4/                    # Validation Framework (Week 4)
├── phase5/                    # Analysis & Final Report (Weeks 5–6)
├── data/                      # NOT committed to git (see .gitignore)
│   ├── corpus/                    # Downloaded dataset
│   └── ground_truth/              # Manually annotated pages (30–50)
└── outputs/                   # Generated charts, stats (committed to git)
```

## Dataset

**Corpus**: [AlbertoChestnut/telugu-ocr](https://huggingface.co/datasets/AlbertoChestnut/telugu-ocr) (teammate-assembled; sourced from Telugu Wikisource, CC BY-SA 4.0)

- ~217 books, ~32,949 aligned image–text page pairs
- Quality labels via Wikisource ProofreadPage (levels 0–4)
- Layout: each book is a folder directly under the corpus root (e.g. `data/corpus/<book_name>/`), containing `meta.json` plus paired `page_NNNN.jpg` / `page_NNNN.txt` files

## Quickstart

Run the numbered scripts in `phase1/` in order:

```bash
# 1. Set up environment
git clone <your-repo-url>
cd telugu-ocr-project
pip install --upgrade huggingface_hub
pip install -r requirements.txt

# 2. Download the corpus (~10.3 GB; needs a free HuggingFace token)
python phase1/0_download_corpus.py

# 3. Build the corpus profile index
python phase1/1_build_profile.py --corpus-dir data/corpus/

# 4. Run characterization (stats + charts → outputs/)
python phase1/2_corpus_characterize.py --profile-json data/corpus/corpus_profile.json --skip-images

# 5. Sample 40 pages for manual ground-truth annotation
python phase1/3_sample_ground_truth.py --corpus-dir data/corpus/ --n 40 --out data/ground_truth/
```

Each script prints a "Next step" hint when it finishes.

## Environment

See `requirements.txt` for pip dependencies.

## Academic Integrity

LLM tools (Claude, Copilot) were used for code scaffolding and documentation. All code has been reviewed and is understood by the authors. See individual file headers for specific attribution.
