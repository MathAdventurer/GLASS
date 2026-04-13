#!/usr/bin/env python3
"""Summarize GLASS two-stage LoRA text-adapter diagnostic JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET_ORDER = [
    "MUTAG", "PROTEINS", "DD", "ENZYMES", "DHFR", "BZR",
    "COX2", "AIDS", "IMDB-BINARY", "NCI1", "COLLAB", "REDDIT-BINARY",
]
DISPLAY_NAMES = {
    "DD": r"D\&D",
    "IMDB-BINARY": "IMDB-B",
    "REDDIT-BINARY": "REDDIT-B",
}


def load_rows(result_root: Path, require_all: bool) -> list[dict]:
    by_dataset = {}
    for path in result_root.glob("*_text_adapter_*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        by_dataset[data["dataset"]] = data
    if not by_dataset:
        raise SystemExit(f"No text-adapter JSON files found under {result_root}")
    missing = [dataset for dataset in DATASET_ORDER if dataset not in by_dataset]
    if require_all and missing:
        raise SystemExit(f"Missing text-adapter results: {', '.join(missing)}")
    return [by_dataset[dataset] for dataset in DATASET_ORDER if dataset in by_dataset]


def value(aggregate: dict, name: str) -> str:
    return f"{aggregate[name + '_mean']:.2f} +/- {aggregate[name + '_std']:.2f}"


def emit_markdown(rows: list[dict]) -> None:
    print("| Dataset | Frozen-stage diagnostic | LoRA tuned | Delta |")
    print("|---|---:|---:|---:|")
    for row in rows:
        aggregate = row["aggregated"]
        print(
            f"| {row['dataset']} | {value(aggregate, 'stage1_auroc')} | "
            f"{value(aggregate, 'auroc')} | {aggregate['delta_mean']:+.2f} |"
        )


def emit_latex(rows: list[dict]) -> None:
    print(r"\begin{tabular}{@{}lccc@{}}")
    print(r"\toprule")
    print(r"Dataset & Frozen-stage diagnostic & LoRA tuned & $\Delta$ \\")
    print(r"\midrule")
    for row in rows:
        aggregate = row["aggregated"]
        name = DISPLAY_NAMES.get(row["dataset"], row["dataset"])
        frozen = value(aggregate, "stage1_auroc").replace("+/-", r"$\pm$")
        tuned = value(aggregate, "auroc").replace("+/-", r"$\pm$")
        print(f"{name} & {frozen} & {tuned} & {aggregate['delta_mean']:+.2f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", type=Path, default=Path("results/text_adapter"))
    parser.add_argument("--format", choices=["markdown", "latex"], default="markdown")
    parser.add_argument("--require_all", action="store_true",
                        help="Fail unless results for all twelve paper datasets are present.")
    args = parser.parse_args()
    rows = load_rows(args.result_root, require_all=args.require_all)
    if args.format == "latex":
        emit_latex(rows)
    else:
        emit_markdown(rows)


if __name__ == "__main__":
    main()
