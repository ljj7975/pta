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


def parse(path: str) -> dict[str, dict[str, float]]:
    data: dict[str, dict[str, float]] = defaultdict(dict)
    with open(path) as f:
        for line in f:
            m = RESULT_RE.match(line.strip())
            if m:
                method, dataset, score = m.group(1), m.group(2), float(m.group(3).rstrip("."))
                data[method][dataset.strip()] = score
    return data


def print_table(data: dict[str, dict[str, float]], methods: list[str] | None) -> None:
    if methods:
        # case-insensitive substring match
        keys = [
            k for k in data
            if any(m.lower() in k.lower() for m in methods)
        ]
    else:
        keys = list(data.keys())

    if not keys:
        print("No matching methods found.")
        return

    # collect all datasets that appear
    all_datasets = sorted({d for k in keys for d in data[k]})
    col_w = 14

    header = f"{'Method':<35}" + "".join(f"{d:>{col_w}}" for d in all_datasets) + f"{'Avg':>{col_w}}"
    print(header)
    print("-" * len(header))

    for method in keys:
        scores = [data[method].get(d) for d in all_datasets]
        valid = [s for s in scores if s is not None]
        avg = sum(valid) / len(valid) if valid else None

        def fmt(s):
            return f"{s:>{col_w}.2f}" if s is not None else f"{'—':>{col_w}}"

        row = f"{method:<35}" + "".join(fmt(s) for s in scores) + fmt(avg)
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="outputs/result.txt")
    parser.add_argument("--methods", nargs="*", help="filter by method name substring")
    args = parser.parse_args()

    data = parse(args.file)
    print_table(data, args.methods)


if __name__ == "__main__":
    main()
