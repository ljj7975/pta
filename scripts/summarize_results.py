#!/usr/bin/env python3
"""
Parse outputs/result.txt and print a per-method table.

Usage:
    python scripts/summarize_results.py
    python scripts/summarize_results.py --file outputs/result.txt --methods exp1 exp2 exp3
"""
import argparse
import re
from collections import defaultdict
from typing import Dict, List, Optional

CD_DATASETS = [
    "caltech101", "dtd", "eurosat", "fgvc",
    "food101", "oxford_flowers", "oxford_pets",
    "stanford_cars", "sun397", "ucf101",
]

# "ClassName's performance on dataset: Top1- 12.34."
RESULT_RE = re.compile(
    r"^(.+?)'s performance on (.+?):\s*Top1-\s*([\d.]+)",
    re.IGNORECASE,
)


def parse(path: str) -> Dict[str, Dict[str, float]]:
    data: Dict[str, Dict[str, float]] = defaultdict(dict)
    with open(path) as f:
        for line in f:
            m = RESULT_RE.match(line.strip())
            if m:
                method, dataset, score = m.group(1), m.group(2), float(m.group(3).rstrip("."))
                data[method][dataset.strip()] = score
    return data


def print_table(data: Dict[str, Dict[str, float]], methods: Optional[List[str]], out_path: str = "outputs/exp_results.txt", ignore_datasets: Optional[List[str]] = None) -> None:
    if methods:
        # case-insensitive substring match
        keys = [
            k for k in data
            if any(m.lower() in k.lower() for m in methods)
        ]
    else:
        keys = list(data.keys())

    if not keys:
        msg = "No matching methods found."
        print(msg)
        with open(out_path, "w") as f:
            f.write(msg + "\n")
        return

    # collect all datasets that appear, minus ignored ones
    ignore = {d.lower() for d in ignore_datasets} if ignore_datasets else set()
    all_datasets = sorted(
        {d for k in keys for d in data[k] if d.lower() not in ignore}
    )
    col_w = 14

    lines: List[str] = []
    header = f"{'Method':<35}" + "".join(f"{d:>{col_w}}" for d in all_datasets) + f"{'Avg':>{col_w}}"
    lines.append(header)
    lines.append("-" * len(header))

    for method in keys:
        scores = [data[method].get(d) for d in all_datasets]
        valid = [s for s in scores if s is not None]
        avg = sum(valid) / len(valid) if valid else None

        def fmt(s):
            return f"{s:>{col_w}.2f}" if s is not None else f"{'—':>{col_w}}"

        row = f"{method:<35}" + "".join(fmt(s) for s in scores) + fmt(avg)
        lines.append(row)

    table_text = "\n".join(lines) + "\n"
    print(table_text, end="")
    with open(out_path, "w") as f:
        f.write(table_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="outputs/result.txt")
    parser.add_argument("--methods", nargs="*", help="filter by method name substring")
    parser.add_argument("--ignore-datasets", nargs="*", help="datasets to exclude from the table (e.g., ucf101 dtd)", default=["ucf101", "caltech101"])
    parser.add_argument("--out", default="outputs/exp_results.txt", help="output file (overwritten)")
    args = parser.parse_args()

    data = parse(args.file)
    print_table(data, args.methods, args.out, args.ignore_datasets)


if __name__ == "__main__":
    main()
