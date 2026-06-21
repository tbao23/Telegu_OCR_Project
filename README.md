# Telugu OCR Project — CSCI/DASC 6020: Machine Learning (Summer 2026)

An end-to-end pipeline for converting scanned Telugu document images into clean Unicode text using Large Language Models, with rigorous LLM-assisted validation.

## Project Structure

```
telugu-ocr-project/
├── README.md
├── requirements.txt
├── .gitignore                     # Excludes data/ from version control
├── phase1/                    # Corpus Characterization (Week 1)
│   ├── run_phase1.py              # Orchestrator — runs steps 0-3 in order
│   ├── 0_download_corpus.py       # Download corpus from HuggingFace
│   ├── 1_build_profile.py         # Build corpus_profile.json from meta.json files
│   ├── 2_corpus_characterize.py   # Profile corpus statistics, generate charts
│   ├── 3_sample_ground_truth.py   # Select 30–50 pages for manual annotation
│   └── phase1_report.qmd          # Written deliverable
├── phase2/                    # Preprocessing Pipeline (Week 2)
├── phase3/                    # OCR Pipeline (Week 3)
├── phase4/                    # Validation Framework (Week 4)
│   ├── run_phase4.py                  # Orchestrator — runs steps 0-5 in order
│   ├── llm_backends.py                # Shared LLM abstraction: ollama/anthropic/openai/gemini
│   ├── 0_make_synthetic_test_data.py  # TEST-ONLY: fake OCR output for dev/testing
│   ├── 1_compute_cer_wer.py           # Classical CER/WER metrics
│   ├── 2_llm_fluency_score.py         # LLM Method A: fluency scoring (backend-configurable)
│   ├── 3_llm_error_detection.py       # LLM Method B: error detection (backend-configurable)
│   ├── 4_cross_model_agreement.py     # LLM Method C: cross-model agreement
│   └── 5_calibration_analysis.py      # Correlates LLM scores against real CER/WER
├── phase5/                    # Analysis & Final Report (Weeks 5–6)
│   ├── run_phase5.py                  # Orchestrator — runs steps 0-1 in order
│   ├── 0_error_categorization.py      # Classifies error types (substitution/diacritic/etc.)
│   └── 1_scalability_cost_estimate.py # Full-corpus time/cost projection
├── data/                      # NOT committed to git (see .gitignore)
│   ├── corpus/                    # Downloaded dataset
│   ├── ground_truth/              # Manually annotated pages (30–50)
│   └── synthetic_ocr_output/      # TEST-ONLY fake OCR output (phase4/0_...)
└── outputs/                   # Generated charts, stats (committed to git)
    ├── phase4/                     # CER/WER, fluency, agreement, calibration results
    └── phase5/                     # Error categories, cost estimates
```

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
| Ollama | Runs Phase 4 LLM validation (Methods A & B) **locally and free** — no API key, no per-token cost | [ollama.com/download](https://ollama.com/download), then `ollama pull qwen3:8b` |

Verify each with:
```bash
python --version
git --version
quarto --version
ollama --version
```

::: {.callout-note}
**Backend is configurable.** Phase 4's LLM validation (fluency scoring,
error detection) defaults to local Ollama — free, no API key, no
per-token cost, at the expense of somewhat weaker Telugu fluency than a
paid model. Pass `--backend anthropic`, `--backend openai`, or
`--backend gemini` to any Phase 4 script (or to `run_phase4.py`) to use
a paid API model as the judge instead — useful for comparing how
different LLM judges agree with each other, or for higher Telugu
accuracy if cost isn't a constraint. See `phase4/llm_backends.py` for
the API key environment variable each backend needs.
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
python phase1/2_corpus_characterize.py --profile-json data/corpus/corpus_profile.json --skip-images
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

## Phase 4/5 Quickstart — Validation & Analysis

These phases need real OCR output from Phase 3. **Until that's ready**,
the orchestrators below work against synthetic test data generated from
the Phase 1 ground truth, so the full pipeline can be built and verified
today.

```bash
# 0. Re-sync dependencies — Phase 4/5 added jiwer, scipy, and requests
#    to requirements.txt. If you set up your environment before these
#    were added, re-run this before continuing:
pip install -r requirements.txt

# 1. One-time: install Ollama and pull the local validation model
#    (see Prerequisites table above for the download link)
ollama pull qwen3:8b
```

**Option A — run everything with two commands:**

```bash
python phase4/run_phase4.py
python phase5/run_phase5.py
```

`run_phase4.py` will ask whether to generate synthetic test data (answer
`n` once you have real Phase 3 output, pointing `--ocr-output-a` at it
instead). It runs CER/WER, both LLM validation methods, cross-model
agreement, and calibration in order. **If the chosen LLM backend isn't
ready (Ollama not running, or an API key missing), it stops immediately
with a clear error** rather than silently skipping — CER/WER results
computed before that point are still saved.

**It's also resumable**: each step skips itself automatically if its
output already exists, so re-running after a partial failure won't
redo expensive LLM calls you've already completed. Use `--force` to
redo every step regardless:

```bash
python phase4/run_phase4.py --backend anthropic    # use Claude as judge instead
python phase4/run_phase4.py --backend ollama --judge-model qwen3:14b
python phase4/run_phase4.py --force                 # redo everything from scratch
```

```bash
python phase4/run_phase4.py --skip-cross-model   # if you only have one model's output
python phase4/run_phase4.py --use-synthetic       # force synthetic data, no prompt
python phase5/run_phase5.py --total-pages 32949   # uses your real corpus size
```

**Option B — run each numbered script individually** (useful for
debugging a single step):

```bash
python phase4/0_make_synthetic_test_data.py --ground-truth data/ground_truth --out data/synthetic_ocr_output/model_a --error-rate 0.08
python phase4/0_make_synthetic_test_data.py --ground-truth data/ground_truth --out data/synthetic_ocr_output/model_b --error-rate 0.15 --seed 99
python phase4/1_compute_cer_wer.py --ground-truth data/ground_truth --ocr-output data/synthetic_ocr_output/model_a --model-name model_a
python phase4/2_llm_fluency_score.py --ocr-output data/synthetic_ocr_output/model_a --model-name model_a
python phase4/3_llm_error_detection.py --ocr-output data/synthetic_ocr_output/model_a --model-name model_a
python phase4/4_cross_model_agreement.py --output-a data/synthetic_ocr_output/model_a --output-b data/synthetic_ocr_output/model_b
python phase4/5_calibration_analysis.py --cer-wer-csv outputs/phase4/cer_wer_model_a.csv --fluency-csv outputs/phase4/fluency_model_a.csv
python phase5/0_error_categorization.py --ground-truth data/ground_truth --ocr-output data/synthetic_ocr_output/model_a --model-name model_a
python phase5/1_scalability_cost_estimate.py --total-pages 32949
```

**Once real Phase 3 OCR output exists**, swap `data/synthetic_ocr_output/model_a`
for the real output folder in any command above — same filenames, same
interface, zero code changes needed.

## Environment

See `requirements.txt` for pip dependencies.

## Academic Integrity

LLM tools (Claude, Copilot) were used for code scaffolding and documentation. All code has been reviewed and is understood by the authors. See individual file headers for specific attribution.
