#!/usr/bin/env python3
"""Analyze scripts/run_trust_reweight_sweep.sh output (Study B).

Parses outputs/result_trust_reweight.txt for TRW-<setting>-s<seed> labels,
computes 4-seed mean accuracy per (setting, dataset) vs. the (1.0,1.0)
control, plus a convergence table (acc_at_fractions, auc_norm, final) pooled
across the 3 datasets x 4 seeds per setting, mirroring Part 4b.5's
methodology. Reports go/no-go verdicts on both final accuracy and
convergence speed, independently per signal_source.

Usage:
    python scripts/analyze_trust_reweight.py \
        --result outputs/result_trust_reweight.txt \
        --records outputs/records_trust_reweight \
        --out outputs/trust_reweight_report.md
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

RESULT_RE = re.compile(r"^(?P<label>.+)'s performance on (?P<dataset>\S+): Top1- (?P<acc>[\d.]+)\.$")
LABEL_RE = re.compile(r"^TRW-(?P<setting>.+)-s(?P<seed>\d+)$")

REGRESSION_BAND = 0.5
GAIN_BAR = 0.3
CONTROL = "control"

SETTING_ORDER = [
    "control",
    "b1.0d0.5-conf", "b1.0d0.1-conf", "b2.0d0.5-conf", "b2.0d0.1-conf",
    "b1.0d0.5-view", "b1.0d0.1-view", "b2.0d0.5-view", "b2.0d0.1-view",
    "b1.0d0.5-both", "b1.0d0.1-both", "b2.0d0.5-both", "b2.0d0.1-both",
]

# matched (boost, boost_down) points, one row per point, one column per source
POINTS = ["b1.0d0.5", "b1.0d0.1", "b2.0d0.5", "b2.0d0.1"]
SOURCES = ["conf", "view", "both"]
SOURCE_LABEL = {"conf": "confusability", "view": "view", "both": "both"}


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


def load_convergence(records_dir, setting, dataset, seed):
    path = Path(records_dir) / f"TRW-{setting}-{dataset}-s{seed}" / "convergence.json"
    if not path.is_file():
        return None
    with open(path) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default="outputs/result_trust_reweight.txt")
    ap.add_argument("--records", default="outputs/records_trust_reweight")
    ap.add_argument("--out", default="outputs/trust_reweight_report.md")
    args = ap.parse_args()

    parsed = parse_results(args.result)
    settings = [s for s in SETTING_ORDER if s in parsed] + sorted(set(parsed) - set(SETTING_ORDER))
    datasets = sorted({d for ds in parsed.values() for d in ds.keys()})
    seeds = sorted({sd for ds in parsed.values() for d in ds.values() for sd in d.keys()})

    means = {s: {d: mean(list(parsed[s].get(d, {}).values())) if parsed[s].get(d) else None for d in datasets} for s in settings}
    control_means = means.get(CONTROL, {})
    control_avg = mean([v for v in control_means.values() if v is not None])

    lines = ["# Study B: Two-Sided Write Reweight by View-Consistency / Prototype-Confusability / Both — Results", ""]
    lines.append(f"Datasets: {', '.join(datasets)} | Seeds: {seeds}")
    lines.append("")
    lines.append("## Overall Accuracy (4-seed mean) vs. `control` (1.0, 1.0)")
    lines.append("")
    lines.append("| Setting | " + " | ".join(datasets) + " | Avg | Δ vs control |")
    lines.append("|" + "---|" * (len(datasets) + 3))

    acc_verdicts = {}
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
            acc_verdicts[s] = (delta > GAIN_BAR and not regressed)

    lines.append("")
    lines.append(f"Go/no-go bar (final accuracy): avg gain > {GAIN_BAR}pp AND no single-dataset regression > {REGRESSION_BAND}pp vs. control.")
    lines.append("(Reference, not rerun: Part 4b.5 patch-vote study never beat control on final accuracy or convergence.)")
    lines.append("")

    # --- Convergence table: pooled across 3 datasets x 4 seeds per setting ---
    lines.append("## Convergence (pooled mean across datasets x seeds)")
    lines.append("")
    lines.append("| Setting | acc@10% | acc@25% | acc@50% | acc@75% | final | auc_norm |")
    lines.append("|---|---|---|---|---|---|---|")

    conv_agg = {}
    for s in settings:
        recs = []
        for d in datasets:
            for sd in seeds:
                c = load_convergence(args.records, s, d, sd)
                if c is not None:
                    recs.append(c)
        if not recs:
            lines.append(f"| {s} | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        agg = {
            "0.1": mean(r["acc_at_fractions"]["0.1"] for r in recs),
            "0.25": mean(r["acc_at_fractions"]["0.25"] for r in recs),
            "0.5": mean(r["acc_at_fractions"]["0.5"] for r in recs),
            "0.75": mean(r["acc_at_fractions"]["0.75"] for r in recs),
            "final": mean(r["final_acc"] for r in recs),
            "auc_norm": mean(r["auc_norm"] for r in recs if r.get("auc_norm") is not None),
        }
        conv_agg[s] = agg
        lines.append(
            f"| {s} | {agg['0.1']:.2f} | {agg['0.25']:.2f} | {agg['0.5']:.2f} | "
            f"{agg['0.75']:.2f} | {agg['final']:.2f} | {agg['auc_norm']:.4f} |"
        )
    lines.append("")

    conv_control = conv_agg.get(CONTROL)
    conv_verdicts = {}
    if conv_control:
        for s in settings:
            if s == CONTROL or s not in conv_agg:
                continue
            improves_early = conv_agg[s]["0.1"] > conv_control["0.1"] + GAIN_BAR
            improves_auc = conv_agg[s]["auc_norm"] > conv_control["auc_norm"] + 0.002
            no_final_cost = conv_agg[s]["final"] >= conv_control["final"] - REGRESSION_BAND
            conv_verdicts[s] = (improves_early or improves_auc) and no_final_cost

    # --- Matched-point comparison: view vs confusability vs both ---
    lines.append("## Matched (boost, boost_down) comparison — Avg accuracy across datasets")
    lines.append("")
    lines.append("| Point | " + " | ".join(SOURCE_LABEL[s] for s in SOURCES) + " |")
    lines.append("|" + "---|" * (len(SOURCES) + 1))
    for point in POINTS:
        row = []
        for src in SOURCES:
            key = f"{point}-{src}"
            if key in means:
                vals = [means[key].get(d) for d in datasets]
                vals = [v for v in vals if v is not None]
                row.append(f"{mean(vals):.3f}" if vals else "N/A")
            else:
                row.append("N/A")
        lines.append(f"| {point} | " + " | ".join(row) + " |")
    lines.append("")

    # --- Verdicts ---
    lines.append("## Promotion Verdict")
    lines.append("")
    all_settings = [s for s in settings if s != CONTROL]
    for s in all_settings:
        acc_go = acc_verdicts.get(s, False)
        conv_go = conv_verdicts.get(s, False)
        if acc_go:
            verdict = "**PROMOTE to held-out validation** (final-accuracy gain)"
        elif conv_go:
            verdict = "no-go on final accuracy, but **partial/mechanistic finding**: faster convergence without cost"
        else:
            verdict = "no-go"
        lines.append(f"- `{s}`: {verdict}")
    lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[OK] report written to {out_path}")


if __name__ == "__main__":
    main()
