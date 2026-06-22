# Phase 5 – Analysis, Insights, and Final Report

## Objective

Interpret the Phase 3/4 results: categorize error types, quantify
preprocessing impact, identify failure modes, and estimate the cost/time
to scale to the full corpus.

## Pipeline Steps

### 1. Error categorization

Classifies OCR errors by type (substitution, deletion, insertion, diacritic
errors, hallucination, truncation) — not just a single CER number.

```
python phase5/1_error_categorization.py --ground-truth data/ground_truth --ocr-output data/synthetic_ocr_output/model_a --model-name model_a
```

### 2. Scalability and cost estimate

Projects time/cost to run each candidate model across the full corpus
(~32,949 pages), using published API pricing and either assumed or
measured per-page timing.

```
python phase5/2_scalability_cost_estimate.py --total-pages 32949
```

### Or run both steps at once:

```
python phase5/run_phase5.py --total-pages 32949
```

## Remaining Manual Work

This phase's automated scripts produce the underlying data; the actual
**final report** (model comparison visualizations, failure-mode discussion,
preprocessing-impact analysis) is written in `phase1/phase1_report.qmd`
(extended to cover all phases) or a new `phase5/final_report.qmd`, pulling
from:

```
outputs/                  Phase 1 corpus characterization
outputs/phase4/           CER/WER, LLM validation, calibration results
outputs/phase5/           Error categories, cost estimate
```

Per the assignment, the final report must be minimum 15 pages (excluding
code/figures), covering methods, results, analysis, and conclusions for
all five phases — rendered to both HTML and PDF via Quarto.

## Inputs

```
data/ground_truth/*.txt            Reference text
data/synthetic_ocr_output/         OCR output (swap for real Phase 3 output)
```

## Outputs

```
outputs/phase5/error_categories_<model>.csv
outputs/phase5/error_categories_<model>_summary.json
outputs/phase5/scalability_cost_estimate.json
outputs/phase5/scalability_cost_estimate.md
```

## Dependencies

pip install jiwer pandas
