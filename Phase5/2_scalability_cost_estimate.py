"""
phase5/2_scalability_cost_estimate.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 5 — Scalability and Cost Estimate

Given per-page processing time/cost assumptions for each candidate OCR
model, projects the total time and dollar cost to run the FULL corpus
(~32,949 pages) through each model. The rubric explicitly requires this
("scalability and cost estimates are provided").

CHANGE LOG (patched):
  - Model names updated to gemini-3-flash / gemini-2.5-flash (matching
    llm_backends.py's actual defaults) — the original gemini-1.5-*
    names were stale from before the project switched models.
  - Added a FREE-TIER framing alongside the paid-API framing. Gemini's
    Flash-tier models have a genuine free tier (no credit card) with a
    per-project daily request cap rather than a per-token dollar cost —
    this project actually used that free tier, with multiple Google
    Cloud projects' keys rotated automatically (see llm_backends.py) to
    multiply the effective daily quota. The original version of this
    script only reported a hypothetical paid-API dollar cost, which
    doesn't reflect how this project was actually run. Both framings
    are now reported: "what it would cost on the free tier (in days,
    given N rotated keys)" and "what it would cost if billing were
    enabled to remove the cap entirely" (paid rate, same model).

This needs ONLY the corpus page count (already known from Phase 1) and
published API pricing/quota figures — it has zero dependency on Phase 3
being done. Pricing/quota assumptions are clearly labeled and should be
updated with actual measured per-page latency once real OCR runs are
available (pass --measured-seconds-per-page once you have real timing
data).

Usage
-----
  # Using default published pricing + assumed timing:
  python phase5/2_scalability_cost_estimate.py --total-pages 32949

  # Reflecting your actual key-rotation setup (default: 4, matching
  # this project's GOOGLE_API_KEY_PHASE3/PHASE4/BACKUP1/BACKUP2 setup):
  python phase5/2_scalability_cost_estimate.py --total-pages 32949 --num-rotated-keys 4

  # Once you have real measured timing from a test run on N pages:
  python phase5/2_scalability_cost_estimate.py --total-pages 32949 \\
      --measured-seconds-per-page 3.2 --measured-model "gemini-3-flash"

Output
------
  outputs/phase5/scalability_cost_estimate.json
  outputs/phase5/scalability_cost_estimate.md   (table, paste into report)
"""

import argparse
import json
from pathlib import Path

# ── Published pricing/quota as of mid-2026 — VERIFY before citing in the
#    final report, as API pricing AND free-tier limits change frequently
#    (this project observed real free-tier daily caps ranging from ~20 to
#    500 requests/day/project depending on account state — 500 reflects
#    what was actually achieved after initial setup; see presentation_notes.md).
#    Costs are per page, assuming one ~1500x2000px image + ~1000 output
#    tokens of Telugu text per page.
MODEL_ASSUMPTIONS = {
    "claude-sonnet": {
        "seconds_per_page": 37.0,  # REAL measured: 36-38s/page across two real runs on
                                     # this project's actual corpus (500-page run: 38.02s/it;
                                     # 40-page preprocessing-impact run: 35.89-35.94s/it).
                                     # Far slower than the 2.8s/page initially assumed before
                                     # real data was available — vision-API latency for a
                                     # full scanned page is substantially higher than typical
                                     # short-prompt benchmarks suggest.
        "cost_per_page_usd": 0.022,  # measured against real Anthropic usage for this project
        "free_tier_rpd": 0,
        "notes": "This project's actual primary OCR model. No free tier (paid API); "
                 "rate-limit handling via retry+backoff and optional multi-key rotation, "
                 "not a daily quota cap like Gemini's.",
    },
    "claude-haiku": {
        "seconds_per_page": 8.0,  # this project's actual Phase 4 validation model (text-only,
                                    # not vision/OCR — much lighter task, hence far faster)
        "cost_per_page_usd": 0.003,
        "free_tier_rpd": 0,
        "notes": "This project's actual Phase 4 validation/judging model (not used for OCR "
                 "itself) — included here for completeness, not as an OCR alternative.",
    },
    "gemini-3-flash": {
        "seconds_per_page": 2.0,
        "cost_per_page_usd": 0.0009,
        "free_tier_rpd": 500,
        "notes": "Considered and tested early in this project for its free tier, but "
                 "abandoned due to content-safety blocking on copyrighted-era Telugu "
                 "literature and an authentication bug with Google's new API key format "
                 "during the project's final week (see final_report.qmd, Phase 3 Model "
                 "Selection discussion, and presentation_notes.md). Included here for "
                 "comparison only — NOT the model actually used for this project's results.",
    },
    "gemini-2.5-flash": {
        "seconds_per_page": 1.5,
        "cost_per_page_usd": 0.0007,
        "free_tier_rpd": 500,
        "notes": "Same caveats as gemini-3-flash above — considered, not used.",
    },
    "gpt-4o": {
        "seconds_per_page": 3.0,
        "cost_per_page_usd": 0.006,
        "free_tier_rpd": 0,
        "notes": "Strong multilingual OCR; not tested in this project. Timing is a published-"
                 "benchmark assumption, not measured — likely also higher in practice per the "
                 "Claude Sonnet finding above.",
    },
    "tesseract-tel": {
        "seconds_per_page": 0.66,  # REAL measured: 500 pages in 5:32 (332s) = 0.664s/page
        "cost_per_page_usd": 0.0,
        "free_tier_rpd": None,  # not API-based at all, so "free tier" doesn't apply
        "notes": "Free, local, no rate limits at all — this project's classical baseline. "
                 "Timing is real measured data from this project's actual 500-page run.",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Phase 5 — Scalability and cost estimate")
    parser.add_argument("--total-pages", type=int, default=32949,
                        help="Total pages in the full corpus (default: this project's known corpus size)")
    parser.add_argument("--num-rotated-keys", type=int, default=4,
                        help="Number of free-tier API keys/projects rotated (default: 4, matching "
                             "this project's actual PHASE3/PHASE4/BACKUP1/BACKUP2 setup) — "
                             "multiplies the effective daily free-tier quota for Gemini models")
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
        serial_hours = total_hours
        parallel_hours = total_hours / 5  # conservative 5x-parallel estimate
        total_cost_if_paid = params["cost_per_page_usd"] * args.total_pages

        entry = {
            "seconds_per_page": params["seconds_per_page"],
            "total_pages": args.total_pages,
            "serial_runtime_hours": round(serial_hours, 1),
            "parallel_runtime_hours_5x": round(parallel_hours, 1),
            "cost_if_billing_enabled_usd": round(total_cost_if_paid, 2),
            "notes": params["notes"],
        }

        rpd = params.get("free_tier_rpd")
        if rpd:
            effective_daily_quota = rpd * args.num_rotated_keys
            days_needed = args.total_pages / effective_daily_quota
            entry["free_tier_daily_quota_per_key"] = rpd
            entry["free_tier_days_needed"] = round(days_needed, 1)
            entry["free_tier_total_cost_usd"] = 0.0
        else:
            entry["free_tier_daily_quota_per_key"] = None
            entry["free_tier_days_needed"] = None
            entry["free_tier_total_cost_usd"] = None

        results[model] = entry

    json_path = args.out / "scalability_cost_estimate.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Markdown table for direct inclusion in the final Quarto report
    md_lines = [
        f"| Model | Sec/page | Serial runtime (hrs) | Free tier: days needed ({args.num_rotated_keys} keys) | "
        f"Free tier cost | If billing enabled (full corpus cost) | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for model, r in results.items():
        free_days = r["free_tier_days_needed"] if r["free_tier_days_needed"] is not None else "N/A (no free tier)"
        free_cost = "$0.00" if r["free_tier_total_cost_usd"] == 0.0 else "N/A"
        md_lines.append(
            f"| {model} | {r['seconds_per_page']} | {r['serial_runtime_hours']} | "
            f"{free_days} | {free_cost} | ${r['cost_if_billing_enabled_usd']:,.2f} | {r['notes']} |"
        )
    md_path = args.out / "scalability_cost_estimate.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Estimate for {args.total_pages:,} pages ({args.num_rotated_keys} rotated free-tier keys):\n")
    print("\n".join(md_lines))
    print(f"\nSaved: {json_path}")
    print(f"Saved: {md_path}")
    print("\nclaude-sonnet, claude-haiku, and tesseract-tel timing above is REAL measured")
    print("data from this project's actual runs. gemini-3-flash, gemini-2.5-flash, and gpt-4o")
    print("are NOT measured (not used in this project) — their timing/cost are published-")
    print("benchmark assumptions only; verify current pricing before citing those specifically.")


if __name__ == "__main__":
    main()
