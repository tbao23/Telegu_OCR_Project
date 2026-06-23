# Phase 5 – Analysis, Insights, and Final Report

## Objective

Interpret the Phase 3/4 results: categorize error types, quantify
preprocessing impact, identify failure modes, and estimate the cost/time
to scale to the full corpus.

## Pipeline Steps

### Recommended: run via the project-root orchestrator (does all 5 phases)

```
python run_all.py --keep-ground-truth --sample-size 500 --validation-limit 100 --models claude,tesseract --force
```

### Or run just Phase 5 on real Phase 3 output:

```
python phase5/run_phase5.py --ground-truth data/ocr_sample --ocr-output outputs/phase3/claude --model-name claude --total-pages 32949
```

### Or run each step individually:

#### 1. Error categorization

Classifies OCR errors by type (substitution, deletion, insertion, diacritic
errors, hallucination, truncation) — not just a single CER number.

```
python phase5/1_error_categorization.py --ground-truth data/ocr_sample --ocr-output outputs/phase3/claude --model-name claude
```

#### 2. Scalability and cost estimate

Projects time/cost to run each candidate model across the full corpus
(~32,949 pages). `claude-sonnet` and `tesseract-tel` rows use this
project's own **real measured timing** (37s/page and 0.66s/page
respectively, from the actual 500-page run) rather than a generic
assumption; Gemini/GPT-4o rows are published-benchmark assumptions only,
since those weren't used for this project's actual results (see
`final_report.qmd`'s Phase 3 Model Selection discussion for why).

```
python phase5/2_scalability_cost_estimate.py --total-pages 32949 --num-rotated-keys 4
```

## Final Report

The actual final report lives at the **project root**:
`final_report.qmd` (not inside `phase5/`, and not an extension of
`phase1/phase1_report.qmd` — those stay separate). It covers all five
phases: methods, results, analysis, conclusions, AI tool usage
disclosure. Render both reports at once with:

```
python generate_reports.py
```

This produces `final_report.html`/`.pdf` and
`phase1/phase1_report.html`/`.pdf`. Per the assignment, the final report
must be a minimum 15 pages excluding code/figures.

## Inputs

```
data/ocr_sample/*.txt                Reference text (large corpus sample)
outputs/phase3/<model>/*.txt         Real OCR output from Phase 3
data/ground_truth/*.txt              Reference text (40-page Phase 1 sample, for the
                                      preprocessing-impact comparison — see phase2/README.md)
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
