# Telugu OCR Project — CSCI/DASC 6020: Machine Learning (Summer 2026)

An end-to-end pipeline for converting scanned Telugu document images into clean Unicode text using Large Language Models, with rigorous LLM-assisted validation.

## Project Structure

```
telugu-ocr-project/
├── README.md
├── requirements.txt
├── environment.yml
├── phase1/                    # Corpus Characterization (Week 1)
│   ├── corpus_characterize.py     # Profile corpus statistics
│   ├── sample_ground_truth.py     # Select 30–50 pages for manual annotation
│   └── phase1_report.qmd          # Written deliverable
├── phase2/                    # Preprocessing Pipeline (Week 2)
├── phase3/                    # OCR Pipeline (Week 3)
├── phase4/                    # Validation Framework (Week 4)
├── phase5/                    # Analysis & Final Report (Weeks 5–6)
├── data/
│   └── ground_truth/          # Manually annotated pages (30–50)
└── outputs/                   # Generated OCR output, reports
```

## Dataset

**Corpus**: [AlbertoChestnut/telugu-ocr](https://huggingface.co/datasets/AlbertoChestnut/telugu-ocr) (teammate-assembled; official professor corpus pending)

- ~25,565 aligned image–text page pairs across 221 books
- Sourced from Telugu Wikisource (CC BY-SA 4.0)
- Quality labels via Wikisource ProofreadPage (levels 0–4)

## Quickstart

```bash
# 1. Clone and set up environment
git clone <your-repo-url>
cd telugu-ocr-project
conda env create -f environment.yml
conda activate telugu-ocr

# 2. Download the corpus (requires ~11 GB disk space)
python phase1/corpus_characterize.py --download

# 3. Run Phase 1 characterization
python phase1/corpus_characterize.py --corpus-dir data/corpus/

# 4. Sample ground truth pages
python phase1/sample_ground_truth.py --corpus-dir data/corpus/ --n 40 --out data/ground_truth/
```

## Environment

See `requirements.txt` for pip dependencies or `environment.yml` for conda.

## Academic Integrity

LLM tools (Claude, Copilot) were used for code scaffolding and documentation. All code has been reviewed and is understood by the authors. See individual file headers for specific attribution.
