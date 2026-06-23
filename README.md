# Telugu OCR Project — CSCI/DASC 6020: Machine Learning (Summer 2026)

An end-to-end pipeline for converting scanned Telugu document images into clean Unicode text using Large Language Models, with rigorous LLM-assisted validation.

## Project Structure

```
telugu-ocr-project/
├── README.md
├── requirements.txt
├── llm_backends.py             # Shared LLM abstraction (ollama/anthropic/openai/gemini) — used by Phase 3 & 4
├── run_all.py                  # Master orchestrator — runs all 5 phases end to end
├── final_report.qmd            # Final project report (all 5 phases)
├── references.bib             # Bibliography for final_report.qmd
├── presentation_notes.md       # Running notes for the recorded presentation
├── .gitignore                  # Excludes data/ from version control
├── phase1/                    # Corpus Characterization (Week 1)
│   ├── run_phase1.py              # Orchestrator — runs steps 0-3 in order
│   ├── 0_download_corpus.py       # Download corpus from HuggingFace
│   ├── 1_build_profile.py         # Build corpus_profile.json from meta.json files
│   ├── 2_corpus_characterize.py   # Profile corpus statistics, generate charts
│   ├── 3_sample_ground_truth.py   # Select 30–50 pages for manual annotation
│   ├── optional_restore_ground_truth.py  # Restore a previously-annotated sample
│   ├── optional_lock_ground_truth.py     # Verify ground truth matches an annotation
│   └── phase1_report.qmd          # Phase 1 written deliverable
├── phase2/                    # Preprocessing Pipeline (Week 2)
│   ├── run_phase2.py               # Orchestrator
│   └── 1_preprocess_images.py     # Image preprocessing (grayscale, denoise, binarize)
├── phase3/                    # OCR Pipeline (Week 3)
│   ├── run_phase3.py              # Orchestrator — chains sampling → preprocessing → OCR → comparison
│   ├── 0_sample_corpus_for_ocr.py # OPTIONAL: sample 100+ pages from the full corpus
│   ├── 1_run_ocr.py               # Run OCR (tesseract / gemini / qwen3vl / easyocr)
│   └── 2_compare_ocr_models.py    # CER/WER comparison against reference text
├── phase4/                    # Validation Framework (Week 4)
│   ├── run_phase4.py                  # Orchestrator — runs steps 0-5 in order
│   ├── 0_make_synthetic_test_data.py  # TEST-ONLY: fake OCR output for dev/testing
│   ├── 1_compute_cer_wer.py           # Classical CER/WER metrics
│   ├── 2_llm_fluency_score.py         # LLM Method A: fluency scoring (backend-configurable)
│   ├── 3_llm_error_detection.py       # LLM Method B: error detection (backend-configurable)
│   ├── 4_cross_model_agreement.py     # LLM Method C: cross-model agreement
│   └── 5_calibration_analysis.py      # Correlates LLM scores against real CER/WER
├── phase5/                    # Analysis & Final Report (Weeks 5–6)
│   ├── run_phase5.py                  # Orchestrator — runs steps 1-2 in order
│   ├── 1_error_categorization.py      # Classifies error types (substitution/diacritic/etc.)
│   └── 2_scalability_cost_estimate.py # Full-corpus time/cost projection
├── data/                      # NOT committed to git (see .gitignore)
│   ├── corpus/                    # Downloaded dataset
│   ├── ground_truth/              # Phase 1's manually annotated 40-page sample
│   ├── ocr_sample/                 # Phase 3's larger corpus-wide sample (100-500+ pages)
│   └── synthetic_ocr_output/      # TEST-ONLY fake OCR output (phase4/0_...)
└── outputs/                   # ALL generated charts/stats/results (committed to git)
    ├── phase1/                     # corpus_stats.json, quality_dist.png, etc.
    ├── phase2/                     # preprocessed_images/, preprocessing_report.csv, sample_comparisons/
    ├── phase3/                     # tesseract/, gemini/, metrics/ (OCR output + comparison)
    ├── phase4/                     # CER/WER, fluency, agreement, calibration results
    └── phase5/                     # Error categories, cost estimates
```

## Phase Documentation

Each phase folder has its own README with that phase's full step-by-step
instructions. This root README covers project-wide setup and a condensed
quickstart; the phase READMEs are the authoritative detail reference.

| Phase | README |
|---|---|
| 1 — Corpus Characterization | [phase1/README.md](phase1/README.md) |
| 2 — Image Preprocessing | [phase2/README.md](phase2/README.md) |
| 3 — OCR Pipeline & Model Comparison | [phase3/README.md](phase3/README.md) |
| 4 — LLM-Assisted Validation | [phase4/README.md](phase4/README.md) |
| 5 — Analysis & Final Report | [phase5/README.md](phase5/README.md) |

## Dataset

**Corpus**: [AlbertoChestnut/telugu-ocr](https://huggingface.co/datasets/AlbertoChestnut/telugu-ocr) (teammate-assembled; sourced from Telugu Wikisource, CC BY-SA 4.0)

- ~217 books, ~32,949 aligned image–text page pairs
- Quality labels via Wikisource ProofreadPage (levels 0–4)
- Layout: each book is a folder directly under the corpus root (e.g. `data/corpus/<book_name>/`), containing `meta.json` plus paired `page_NNNN.jpg` / `page_NNNN.txt` files

## Prerequisites

These are **not** installable via `requirements.txt` (pip only manages
Python packages) — install them separately before running anything:

| Tool | Why it's needed | Install |
|------|-----------------|---------|
| Python 3.11+ | Runs all scripts | [python.org](https://python.org) — check "Add to PATH" |
| Git | Clone/version the repo | [git-scm.com](https://git-scm.com) |
| Quarto | Renders `phase1_report.qmd` to HTML/PDF | [quarto.org/docs/get-started](https://quarto.org/docs/get-started) (already bundled with RStudio if installed) |
| TinyTeX | Required only for **PDF** output | After installing Quarto, run: `quarto install tinytex` |
| Ollama | Optional — fully local/free fallback for Phase 3 (`qwen3vl`) and Phase 4 validation. Not the default for either; this project's actual results use Claude/Anthropic instead (see below) | [ollama.com/download](https://ollama.com/download), then `ollama pull qwen3:8b` and `ollama pull qwen3-vl:8b` |
| **Anthropic API key** | **This project's actual default** for Phase 3 OCR (`claude`) and Phase 4 validation. Paid, but cheap at this project's scale (~$13 total measured cost for 580 OCR calls + 200 validation calls — see `phase5/2_scalability_cost_estimate.py`) | [console.anthropic.com](https://console.anthropic.com) — create a key, then `set ANTHROPIC_API_KEY=sk-ant-...` (Windows cmd) or `$env:ANTHROPIC_API_KEY="sk-ant-..."` (PowerShell) |
| Tesseract OCR + Telugu pack | Phase 3's free baseline OCR engine | `apt install tesseract-ocr tesseract-ocr-tel` (Linux) or [github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki) (Windows installer) |

Verify each with:
```bash
python --version
git --version
quarto --version
ollama --version
tesseract --version
```

::: {.callout-note}
**Backend is configurable.** Phase 3's OCR (`1_run_ocr.py`) and Phase 4's
LLM validation both default to **Anthropic (Claude)** — this is what this
project's actual reported results used, after Gemini's free tier turned
out to have real operational problems (content-safety blocking, an API
key authentication bug) during development — see `final_report.qmd`'s
Phase 3 Model Selection discussion. Pass `--backend ollama`, `--backend
openai`, or `--backend gemini` to any Phase 4 script (or `--model
qwen3vl`/`gemini` to Phase 3's `1_run_ocr.py`) to use a different backend
instead. See `llm_backends.py` (project root) for the API key environment
variable each backend needs, including a multi-key rotation setup for
handling rate limits at scale.
:::

## Quickstart

**Option A — run everything with one command:**

```bash
git clone <your-repo-url>
cd telugu-ocr-project
pip install --upgrade huggingface_hub
pip install -r requirements.txt

python phase1/run_phase1.py
```

This runs steps 0–3 in order. It will first ask **"Download the corpus
now? [y/n]"** — answer `n` if you already have it (e.g. from a previous
run), or `y` to download fresh. It then prompts once for your HuggingFace
token if downloading. To skip the prompts entirely:

```bash
python phase1/run_phase1.py --skip-download   # force-skip, no prompt
python phase1/run_phase1.py --download        # force-download, no prompt
```

**Option B — run each numbered script individually** (useful for
debugging a single step, or re-running just one):

```bash
python phase1/0_download_corpus.py
python phase1/1_build_profile.py --corpus-dir data/corpus/
python phase1/2_corpus_characterize.py --profile-json data/corpus/corpus_profile.json
python phase1/3_sample_ground_truth.py --corpus-dir data/corpus/ --n 40 --out data/ground_truth/
```

Either way, the two steps that remain manual are:
1. **Annotation** — open `data/ground_truth/annotation_template.csv` and
   compare each sampled page's transcription against its scan image.
2. **Render the report** — once annotation is done:
   ```bash
   quarto render phase1/phase1_report.qmd
   ```

Each script prints a "Next step" hint when it finishes.

## Full Pipeline Quickstart — Phases 2 through 5

**This is now the recommended way to run everything past Phase 1**, via
the project-root orchestrator:

```bash
python run_all.py --keep-ground-truth --sample-size 500 --validation-limit 100 --models claude,tesseract --force
```

What this single command does, in order:
1. Prompts whether to create a new Phase 1 ground-truth sample, or keep
   your existing one (`--keep-ground-truth` skips the prompt)
2. Preprocesses the 40-page ground truth (Phase 2)
3. Runs real OCR on both the raw and preprocessed versions of those same
   40 pages, with both models — the quantified "does preprocessing
   actually help" comparison the assignment requires
4. Samples 500 fresh pages from the full corpus, preprocesses them, runs
   OCR with both models, compares against reference text (Phase 3)
5. Runs CER/WER, both LLM validation methods (capped at 100 pages via
   `--validation-limit`), cross-model agreement, and calibration (Phase 4)
6. Runs error categorization and the scalability/cost estimate (Phase 5)

**Defaults to Claude + Tesseract** (`--models claude,tesseract`) — make
sure `ANTHROPIC_API_KEY` is set first (see Prerequisites above). Every
step is resumable: if interrupted, re-running the identical command picks
up where it left off, including per-page resumability for OCR calls, so
an interruption never forces you to re-pay for already-completed pages.

For a quick, cheap smoke test before committing to the full 500-page run:
```bash
python run_all.py --keep-ground-truth --sample-size 3 --validation-limit 2 --models claude,tesseract --force
```

### Render the reports once the pipeline has run

```bash
python generate_reports.py
```

Renders both `phase1/phase1_report.qmd` and `final_report.qmd` to HTML
and PDF via Quarto. Use `--only phase1` or `--only final` to render just
one.

### Running phases individually (useful for debugging one step)

```bash
python phase2/run_phase2.py --force
python phase3/run_phase3.py --use-corpus-sample --sample-size 500 --models claude,tesseract
python phase4/run_phase4.py --ground-truth data/ocr_sample --ocr-output-a outputs/phase3/claude --model-name-a claude --ocr-output-b outputs/phase3/tesseract --model-name-b tesseract --skip-synthetic --validation-limit 100
python phase5/run_phase5.py --ground-truth data/ocr_sample --ocr-output outputs/phase3/claude --model-name claude --total-pages 32949
```

See each phase's own README (`phase2/README.md` through `phase5/README.md`)
for the full breakdown of every individual script within each phase, and
for the (optional) synthetic-test-data path useful for development without
spending API calls.

## Environment

See `requirements.txt` for pip dependencies.

## Academic Integrity

LLM tools (Claude, Copilot) were used for code scaffolding and documentation. All code has been reviewed and is understood by the authors. See individual file headers for specific attribution.
