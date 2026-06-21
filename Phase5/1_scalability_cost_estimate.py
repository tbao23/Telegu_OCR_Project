"""
phase5/1_scalability_cost_estimate.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 5, Step 1 — Scalability and Cost Estimate

Given per-page processing time/cost assumptions for each candidate OCR
model, projects the total time and dollar cost to run the FULL corpus
(~32,949 pages) through each model. The rubric explicitly requires this
("scalability and cost estimates are provided").

This needs ONLY the corpus page count (already known from Phase 1) and
published API pricing — it has zero dependency on Phase 3 being done.
Pricing assumptions are clearly labeled and should be updated with
actual measured per-page latency once real OCR runs are available
(pass --measured-seconds-per-page once you have real timing data).

Usage
-----
  # Using default published pricing + assumed timing:
  python phase5/1_scalability_cost_estimate.py --total-pages 32949

  # Once you have real measured timing from a test run on N pages:
  python phase5/1_scalability_cost_estimate.py --total-pages 32949 \\
      --measured-seconds-per-page 3.2 --measured-model "gemini-1.5-pro"

Output
------
  outputs/phase5/scalability_cost_estimate.json
  outputs/phase5/scalability_cost_estimate.md   (table, paste into report)
"""

import argparse
import json
from pathlib import Path

# ── Published pricing as of early 2026 — VERIFY before citing in the final
#    report, as API pricing changes frequently. Costs are per page, assuming
#    one ~1500x2000px image + ~1000 output tokens of Telugu text per page.
MODEL_ASSUMPTIONS = {
    "gemini-1.5-flash": {
        "seconds_per_page": 1.5,
        "cost_per_page_usd": 0.0007,
        "notes": "Cheapest viable option; good Indic script support",
    },
    "gemini-1.5-pro": {
        "seconds_per_page": 2.5,
        "cost_per_page_usd": 0.0035,
        "notes": "Best published Indic-script accuracy among API models",
    },
    "gpt-4o": {
        "seconds_per_page": 3.0,
        "cost_per_page_usd": 0.006,
        "notes": "Strong multilingual OCR; higher cost than Gemini",
    },
    "claude-sonnet": {
        "seconds_per_page": 2.8,
        "cost_per_page_usd": 0.005,
        "notes": "Good vision + multilingual; similar cost tier to GPT-4o",
    },
    "tesseract-tel": {
        "seconds_per_page": 0.8,
        "cost_per_page_usd": 0.0,
        "notes": "Free, local, but lower accuracy baseline — no API cost or rate limits",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Phase 5 — Scalability and cost estimate")
    parser.add_argument("--total-pages", type=int, default=32949,
                        help="Total pages in the full corpus (default: this project's known corpus size)")
    parser.add_argument("--measured-seconds-per-page", type=float, default=None,
                        help="If you have real timing data from a test run, override the assumption")
    parser.add_argument("--measured-model", type=str, default=None,
                        help="Which model the measured timing applies to")
    parser.add_argument("--out", type=Path, default=Path("outputs/phase5"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    assumptions = {k: dict(v) for k, v in MODEL_ASSUMPTIONS.items()}
    if args.measured_seconds_per_page and args.measured_model in assumptions:
        assumptions[args.measured_model]["seconds_per_page"] = args.measured_seconds_per_page
        assumptions[args.measured_model]["notes"] += " (timing updated from real measurement)"

    results = {}
    for model, params in assumptions.items():
        total_seconds = params["seconds_per_page"] * args.total_pages
        total_hours = total_seconds / 3600
        # Assume processing can run unattended; report both serial and a
        # conservative 5x-parallel estimate (typical free-tier API concurrency)
        serial_hours = total_hours
        parallel_hours = total_hours / 5
        total_cost = params["cost_per_page_usd"] * args.total_pages

        results[model] = {
            "seconds_per_page": params["seconds_per_page"],
            "total_pages": args.total_pages,
            "serial_runtime_hours": round(serial_hours, 1),
            "parallel_runtime_hours_5x": round(parallel_hours, 1),
            "total_cost_usd": round(total_cost, 2),
            "notes": params["notes"],
        }

    json_path = args.out / "scalability_cost_estimate.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Markdown table for direct inclusion in the final Quarto report
    md_lines = [
        "| Model | Sec/page | Serial runtime (hrs) | 5x-parallel runtime (hrs) | Total cost (full corpus) | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for model, r in results.items():
        md_lines.append(
            f"| {model} | {r['seconds_per_page']} | {r['serial_runtime_hours']} | "
            f"{r['parallel_runtime_hours_5x']} | ${r['total_cost_usd']:,.2f} | {r['notes']} |"
        )
    md_path = args.out / "scalability_cost_estimate.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Estimate for {args.total_pages:,} pages:\n")
    print("\n".join(md_lines))
    print(f"\nSaved: {json_path}")
    print(f"Saved: {md_path}")
    print("\n(Pricing assumptions are placeholders — verify current API pricing")
    print(" and re-run with --measured-seconds-per-page once real timing is available.)")


if __name__ == "__main__":
    main()
