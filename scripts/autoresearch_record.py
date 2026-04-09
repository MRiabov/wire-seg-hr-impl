#!/usr/bin/env python3
"""Append a benchmark result row to results.tsv.

The benchmark JSON is produced by scripts/autoresearch_verify.sh / infer.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


HEADER = [
    "commit",
    "fine_iou",
    "fine_f1",
    "coarse_iou",
    "coarse_f1",
    "avg_coarse_ms",
    "avg_fine_ms",
    "avg_total_ms",
    "peak_allocated_gb",
    "status",
    "description",
]


def _git_commit_short(root: Path) -> str:
    out = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True
    )
    return out.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        default=".autoresearch/bench.json",
        help="Path to benchmark JSON produced by infer.py",
    )
    parser.add_argument(
        "--results",
        default="results.tsv",
        help="TSV ledger to append to",
    )
    parser.add_argument(
        "--status",
        default="keep",
        choices=["keep", "discard", "crash"],
        help="Experiment outcome",
    )
    parser.add_argument(
        "--description",
        default="",
        help="Short description of what changed",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = root / report_path
    results_path = Path(args.results)
    if not results_path.is_absolute():
        results_path = root / results_path

    if not report_path.is_file():
        raise SystemExit(f"Missing benchmark report: {report_path}")

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    summary = report.get("summary", {})

    row = {
        "commit": _git_commit_short(root),
        "fine_iou": summary.get("fine_iou", 0.0),
        "fine_f1": summary.get("fine_f1", 0.0),
        "coarse_iou": summary.get("coarse_iou", 0.0),
        "coarse_f1": summary.get("coarse_f1", 0.0),
        "avg_coarse_ms": summary.get("avg_coarse_ms", 0.0),
        "avg_fine_ms": summary.get("avg_fine_ms", 0.0),
        "avg_total_ms": summary.get("avg_ms", 0.0),
        "peak_allocated_gb": summary.get("peak_allocated_gb", 0.0),
        "status": args.status,
        "description": args.description.replace("\t", " ").replace("\n", " "),
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_path.exists()
    with results_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[autoresearch] appended {results_path}")


if __name__ == "__main__":
    main()
