#!/usr/bin/env python3
"""Analyze scripts/run_confusability_sweep.sh output (Phase 2b).

Parses outputs/result_confusability.txt for CGP-<setting>-s<seed> labels,
computes 4-seed mean accuracy per (setting, dataset), and reports how many
classes end up frozen at the end of each run (from
outputs/records_confusability/**/records.jsonl's per-sample n_frozen field)
to explain the mechanism behind any result.

Usage:
    python scripts/analyze_confusability.py \
        --result-file outputs/result_confusability.txt \
        --record-root outputs/records_confusability \
        --out outputs/confusability_report.md
"""
import argparse
import json
import os
import re
from collections import defaultdict

LINE_RE = re.compile(r"^(CGP-[\w.]+)-s(\d+)'s performance on (\S+): Top1- ([\d.]+)\.$")


def parse_results(path):
    # (setting, dataset) -> {seed: acc}
    data = defaultdict(dict)
    with open(path) as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if not m:
                continue
            label, seed, dataset, acc = m.groups()
            setting = label[len("CGP-"):]
            data[(setting, dataset)][int(seed)] = float(acc)
    return data


def final_n_frozen(record_root, setting, dataset, seed):
    path = os.path.join(record_root, f"CGP-{setting}-{dataset}-s{seed}", "records.jsonl")
    if not os.path.exists(path):
        return None
    last = None
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if not obj.get("__header__"):
                last = obj
    return last.get("n_frozen") if last else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-file", default="outputs/result_confusability.txt")
    ap.add_argument("--record-root", default="outputs/records_confusability")
    ap.add_argument("--datasets", default="dtd/oxford_flowers/oxford_pets")
    ap.add_argument("--seeds", default="1,2,3,4")
    ap.add_argument("--out", default="outputs/confusability_report.md")
    args = ap.parse_args()

    datasets = args.datasets.split("/")
    seeds = [int(s) for s in args.seeds.split(",")]
    data = parse_results(args.result_file)

    settings = sorted({s for (s, d) in data.keys()}, key=lambda s: (s != "control", s))

    lines = ["# Phase 2b: Prototype Confusability Write-Freeze -- Results", ""]
    lines.append(f"Datasets: {', '.join(datasets)} | Seeds: {seeds}")
    lines.append("")
    lines.append("## Overall Accuracy (4-seed mean) vs. Control")
    lines.append("")
    header = "| Setting | " + " | ".join(datasets) + " | Avg | Delta Avg vs control |"
    lines.append(header)
    lines.append("|" + "---|" * (len(datasets) + 3))

    means = {}
    for setting in settings:
        vals = []
        for dataset in datasets:
            accs = [data[(setting, dataset)][s] for s in seeds if s in data[(setting, dataset)]]
            vals.append(sum(accs) / len(accs) if accs else float("nan"))
        avg = sum(vals) / len(vals)
        means[setting] = (vals, avg)

    control_avg = means["control"][1]
    go_no_go = {}
    for setting in settings:
        vals, avg = means[setting]
        delta = avg - control_avg
        row = f"| {setting} | " + " | ".join(f"{v:.2f}" for v in vals) + f" | {avg:.3f} | {delta:+.3f} |"
        lines.append(row)
        if setting != "control":
            control_vals = means["control"][0]
            per_ds_delta = [v - cv for v, cv in zip(vals, control_vals)]
            regress = any(d < -0.5 for d in per_ds_delta)
            go_no_go[setting] = (delta > 0.3 and not regress)

    lines.append("")
    lines.append("Go/no-go bar: avg gain > 0.3pp AND no single-dataset regression > 0.5pp vs. control.")
    lines.append("")
    lines.append("## Promotion Verdict")
    lines.append("")
    for setting, ok in go_no_go.items():
        lines.append(f"- `{setting}`: **{'go' if ok else 'no-go'}**")
    lines.append("")

    lines.append("## Final n_frozen (out of C classes) -- pooled mean across seeds")
    lines.append("")
    header2 = "| Setting | " + " | ".join(datasets) + " |"
    lines.append(header2)
    lines.append("|" + "---|" * (len(datasets) + 1))
    for setting in settings:
        row_vals = []
        for dataset in datasets:
            vals = [final_n_frozen(args.record_root, setting, dataset, s) for s in seeds]
            vals = [v for v in vals if v is not None]
            row_vals.append(f"{sum(vals)/len(vals):.1f}" if vals else "N/A")
        lines.append(f"| {setting} | " + " | ".join(row_vals) + " |")
    lines.append("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[OK] report written to {args.out}")


if __name__ == "__main__":
    main()
