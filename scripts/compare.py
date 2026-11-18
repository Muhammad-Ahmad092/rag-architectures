"""Aggregate every ``*/eval/results.json`` (fair mode only) into the root README table.

    python scripts/compare.py [--root PATH]

Rewrites the block between ``<!-- EVAL_TABLE -->`` and ``<!-- /EVAL_TABLE -->``.
Only standard-library imports, so it runs anywhere.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
START, END = "<!-- EVAL_TABLE -->", "<!-- /EVAL_TABLE -->"


def collect(root: Path) -> list[dict]:
    results = []
    for path in sorted(root.glob("*/eval/results.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("mode") == "fair":
            results.append(data)
    return results


def table(results: list[dict]) -> str:
    if not results:
        return "_No results yet._"
    header = "| RAG type | " + " | ".join(METRICS) + " | skipped |"
    sep = "|" + "---|" * (len(METRICS) + 2)
    rows = []
    for r in results:
        cells = [f"{r['metrics'][m]:.3f}" if r["metrics"].get(m) is not None else "–" for m in METRICS]
        rows.append(f"| [{r['folder']}]({r['folder']}/) | " + " | ".join(cells) + f" | {len(r.get('skipped', []))} |")
    return "\n".join([header, sep, *rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    readme = args.root / "README.md"
    text = readme.read_text(encoding="utf-8")
    block = f"{START}\n{table(collect(args.root))}\n{END}"
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    readme.write_text(new, encoding="utf-8")
    print(f"Updated {readme}")


if __name__ == "__main__":
    main()
