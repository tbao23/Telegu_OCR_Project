"""
generate_reports.py  (project root)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Renders both project reports (Phase 1 and the Final Report) to HTML
and PDF via Quarto, in one command.

Requires Quarto installed and on PATH (https://quarto.org/docs/get-started/),
plus TinyTeX for PDF output (one-time setup: `quarto install tinytex`).

Usage
-----
  python generate_reports.py

  # Only render one of the two:
  python generate_reports.py --only phase1
  python generate_reports.py --only final

Output
------
  phase1/phase1_report.html
  phase1/phase1_report.pdf
  final_report.html
  final_report.pdf
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent

REPORTS = {
    "phase1": ROOT_DIR / "phase1" / "phase1_report.qmd",
    "final": ROOT_DIR / "final_report.qmd",
}


def check_quarto_ready() -> None:
    if not shutil.which("quarto"):
        sys.exit(
            "ERROR: 'quarto' command not found on PATH.\n"
            "  Install: https://quarto.org/docs/get-started/\n"
            "  (If installed via RStudio, you may need to add its quarto/bin "
            "folder to PATH manually — see project README.)"
        )


def render(qmd_path: Path) -> bool:
    if not qmd_path.exists():
        print(f"\n[SKIP] {qmd_path} not found.")
        return False

    print(f"\n{'=' * 70}")
    print(f"  Rendering: {qmd_path}")
    print(f"{'=' * 70}")

    result = subprocess.run(["quarto", "render", str(qmd_path)])

    if result.returncode != 0:
        print(f"\n[FAILED] {qmd_path} — see Quarto's error output above.")
        print("  Common causes: missing TinyTeX (`quarto install tinytex`), a Python")
        print("  package import error in a code chunk, or a referenced output file")
        print("  (chart/CSV/JSON) that doesn't exist yet — render after the relevant")
        print("  pipeline phase has actually been run.")
        return False

    html_path = qmd_path.with_suffix(".html")
    pdf_path = qmd_path.with_suffix(".pdf")
    print(f"\n[OK] Rendered:")
    if html_path.exists():
        print(f"  {html_path}")
    if pdf_path.exists():
        print(f"  {pdf_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Render both project reports via Quarto")
    parser.add_argument("--only", choices=["phase1", "final"], default=None,
                        help="Render only one report instead of both")
    args = parser.parse_args()

    check_quarto_ready()

    targets = [args.only] if args.only else list(REPORTS.keys())

    results = {}
    for name in targets:
        results[name] = render(REPORTS[name])

    print(f"\n{'=' * 70}")
    print("  Summary")
    print(f"{'=' * 70}")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED or SKIPPED"
        print(f"  {name}: {status}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
