#!/usr/bin/env python3
"""Analyze scripts/run_trust_write_gate_sweep.sh output (Study A).

Parses outputs/result_trust_write_gate.txt for TWG-<setting>-s<seed> labels,
computes 4-seed mean accuracy per (setting, dataset) vs. baseline, and
reports the write rate per setting (from the write_occurred record field)
to explain any result mechanistically, mirroring Part 4a.3's methodology.

Usage:
    python scripts/analyze_trust_write_gate.py \
        --result outputs/result_trust_write_gate.txt \
        --records outputs/records_trust_write_gate \
        --out outputs/trust_write_gate_report.md
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

RESULT_RE = re.compile(r"^(?P<label>.+)'s performance on (?P<dataset>\S+): Top1- (?P<acc>[\d.]+)\.$")
LABEL_RE = re.compile(r"^TWG-(?P<setting>.+)-s(?P<seed>\d+)$")

REGRESSION_BAND = 0.5
GAIN_BAR = 0.3
CONTROL = "baseline"

SETTING_ORDER = ["baseline", "clean-conf", "cac-conf", "clean-view", "cac-view", "clean-both", "cac-both"]


def parse_results(path):
    out = defaultdict(lambda: defaultdict(dict))
    with open(path) as fh:
        for line in fh:
            m = RESULT_RE.match(line.strip())
            if not m:
                continue
            lm = LABEL_RE.match(m.group("label"))
            if not lm:
                continue
            out[lm.group("setting")][m.group("dataset")][int(lm.group("seed"))] = float(m.group("acc"))
    return out


def write_rate(records_dir, setting, dataset, seed):
    path = Path(records_dir) / f"TWG-{setting}-{dataset}-s{seed}" / "records.jsonl"
    if not path.is_file():
        return None
    n_open = n_total = 0
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("__header__"):
                continue
            n_total += 1
            if rec.get("write_occurred"):
                n_open += 1
    return (n_open / n_total) if n_total else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default="outputs/result_trust_write_gate.txt")
    ap.add_argument("--records", default="outputs/records_trust_write_gate")
    ap.add_argument("--out", default="outputs/trust_write_gate_report.md")
    args = ap.parse_args()

    parsed = parse_results(args.result)
    settings = [s for s in SETTING_ORDER if s in parsed] + sorted(set(parsed) - set(SETTING_ORDER))
    datasets = sorted({d for ds in parsed.values() for d in ds.keys()})
    seeds = sorted({sd for ds in parsed.values() for d in ds.values() for sd in d.keys()})

    means = {s: {d: mean(list(parsed[s].get(d, {}).values())) if parsed[s].get(d) else None for d in datasets} for s in settings}
    control_means = means.get(CONTROL, {})
    control_avg = mean([v for v in control_means.values() if v is not None])

    lines = ["# Study A: Write-Gate by View-Consistency / Prototype-Confusability / Both — Results", ""]
    lines.append(f"Datasets: {', '.join(datasets)} | Seeds: {seeds}")
    lines.append("")
    lines.append("## Overall Accuracy (4-seed mean) vs. `baseline`")
    lines.append("")
    lines.append("| Setting | " + " | ".join(datasets) + " | Avg | Δ vs baseline |")
    lines.append("|" + "---|" * (len(datasets) + 3))

    verdicts = {}
    for s in settings:
        vals = [means[s].get(d) for d in datasets]
        avg = mean([v for v in vals if v is not None])
        delta = avg - control_avg
        cells = [f"{v:.2f}" if v is not None else "N/A" for v in vals]
        lines.append(f"| {s} | " + " | ".join(cells) + f" | {avg:.3f} | {delta:+.3f} |")
        if s != CONTROL:
            regressed = any(
                (means[s].get(d) is not None and control_means.get(d) is not None
                 and (means[s][d] - control_means[d]) < -REGRESSION_BAND)
                for d in datasets
            )
            verdicts[s] = (delta > GAIN_BAR and not regressed)

    lines.append("")
    lines.append(f"Go/no-go bar: avg gain > {GAIN_BAR}pp AND no single-dataset regression > {REGRESSION_BAND}pp vs. baseline.")
    lines.append("(Reference, not rerun: Part 4a `confident` — dtd 42.48 / flowers 74.12 / pets 90.36.)")
    lines.append("")
    lines.append("## Promotion Verdict")
    lines.append("")
    for s in settings:
        if s == CONTROL:
            continue
        lines.append(f"- `{s}`: **{'PROMOTE to held-out validation' if verdicts[s] else 'no-go'}**")
    lines.append("")

    lines.append("## Write Rate (pooled mean across seeds; fraction of samples where the gate fired)")
    lines.append("")
    lines.append("| Setting | " + " | ".join(datasets) + " |")
    lines.append("|" + "---|" * (len(datasets) + 1))
    for s in settings:
        row = []
        for d in datasets:
            rates = [write_rate(args.records, s, d, sd) for sd in seeds]
            rates = [r for r in rates if r is not None]
            row.append(f"{mean(rates):.3f}" if rates else "N/A")
        lines.append(f"| {s} | " + " | ".join(row) + " |")
    lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[OK] report written to {out_path}")


if __name__ == "__main__":
    main()
