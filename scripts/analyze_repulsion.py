#!/usr/bin/env python3
"""Analyze scripts/run_repulsion_sweep.sh output.

Parses outputs/result_repulsion.txt for REP-<setting>-s<seed> labels,
computes 4-seed mean accuracy per (setting, dataset) vs. control
(repulsion_lr=0.0), and reports the mechanism evidence this study is
actually testing: repulsion fire rate, and whether the correction reduces
nearest-other-class confusability (target_confusability pre- vs.
post_repulsion_confusability), independent of whether accuracy moved.

Usage:
    python scripts/analyze_repulsion.py \
        --result outputs/result_repulsion.txt \
        --records outputs/records_repulsion \
        --out outputs/repulsion_report.md
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

RESULT_RE = re.compile(r"^(?P<label>.+)'s performance on (?P<dataset>\S+): Top1- (?P<acc>[\d.]+)\.$")
LABEL_RE = re.compile(r"^REP-(?P<setting>.+)-s(?P<seed>\d+)$")

REGRESSION_BAND = 0.5
GAIN_BAR = 0.3
CONTROL = "control"

SETTING_ORDER = [
    "control",
    "lr0.02-conf", "lr0.05-conf", "lr0.1-conf",
    "lr0.02-always", "lr0.05-always", "lr0.1-always",
]


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


def mechanism_stats(records_dir, setting, dataset, seed):
    """Return (n_total, n_applied, mean_pre_when_applied, mean_post_when_applied)."""
    path = Path(records_dir) / f"REP-{setting}-{dataset}-s{seed}" / "records.jsonl"
    if not path.is_file():
        return None
    n_total = n_applied = 0
    pre_vals, post_vals = [], []
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("__header__"):
                continue
            n_total += 1
            if rec.get("repulsion_applied"):
                n_applied += 1
                if rec.get("target_confusability") is not None:
                    pre_vals.append(rec["target_confusability"])
                if rec.get("post_repulsion_confusability") is not None:
                    post_vals.append(rec["post_repulsion_confusability"])
    return {
        "n_total": n_total,
        "n_applied": n_applied,
        "fire_rate": (n_applied / n_total) if n_total else None,
        "mean_pre": mean(pre_vals) if pre_vals else None,
        "mean_post": mean(post_vals) if post_vals else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default="outputs/result_repulsion.txt")
    ap.add_argument("--records", default="outputs/records_repulsion")
    ap.add_argument("--out", default="outputs/repulsion_report.md")
    args = ap.parse_args()

    parsed = parse_results(args.result)
    settings = [s for s in SETTING_ORDER if s in parsed] + sorted(set(parsed) - set(SETTING_ORDER))
    datasets = sorted({d for ds in parsed.values() for d in ds.keys()})
    seeds = sorted({sd for ds in parsed.values() for d in ds.values() for sd in d.keys()})

    means = {s: {d: mean(list(parsed[s].get(d, {}).values())) if parsed[s].get(d) else None for d in datasets} for s in settings}
    control_means = means.get(CONTROL, {})
    control_avg = mean([v for v in control_means.values() if v is not None])

    lines = ["# Phase 8: Prototype Repulsion — Results", ""]
    lines.append(f"Datasets: {', '.join(datasets)} | Seeds: {seeds}")
    lines.append("")
    lines.append("## Overall Accuracy (4-seed mean) vs. `control` (repulsion_lr=0.0)")
    lines.append("")
    lines.append("| Setting | " + " | ".join(datasets) + " | Avg | Δ vs control |")
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
    lines.append(f"Go/no-go bar: avg gain > {GAIN_BAR}pp AND no single-dataset regression > {REGRESSION_BAND}pp vs. control.")
    lines.append("")

    # --- Mechanism evidence: fire rate + confusability reduction ---
    lines.append("## Mechanism: repulsion fire rate and confusability reduction (pooled across datasets x seeds)")
    lines.append("")
    lines.append("| Setting | Fire rate | Mean pre-confusability | Mean post-confusability | Δ |")
    lines.append("|---|---|---|---|---|")
    for s in settings:
        if s == CONTROL:
            lines.append(f"| {s} | 0.000 (no-op) | N/A | N/A | N/A |")
            continue
        stats_list = []
        for d in datasets:
            for sd in seeds:
                st = mechanism_stats(args.records, s, d, sd)
                if st is not None:
                    stats_list.append(st)
        if not stats_list:
            lines.append(f"| {s} | N/A | N/A | N/A | N/A |")
            continue
        total_n = sum(st["n_total"] for st in stats_list)
        total_applied = sum(st["n_applied"] for st in stats_list)
        fire_rate = total_applied / total_n if total_n else None
        pre_vals = [st["mean_pre"] for st in stats_list if st["mean_pre"] is not None]
        post_vals = [st["mean_post"] for st in stats_list if st["mean_post"] is not None]
        mean_pre = mean(pre_vals) if pre_vals else None
        mean_post = mean(post_vals) if post_vals else None
        delta_str = f"{(mean_post - mean_pre):+.4f}" if (mean_pre is not None and mean_post is not None) else "N/A"
        pre_str = f"{mean_pre:.4f}" if mean_pre is not None else "N/A"
        post_str = f"{mean_post:.4f}" if mean_post is not None else "N/A"
        lines.append(f"| {s} | {fire_rate:.3f} | {pre_str} | {post_str} | {delta_str} |")
    lines.append("")
    lines.append(
        "A negative Δ confirms the repulsion correction mechanically reduces nearest-other-class "
        "similarity for the classes it fires on, independent of whether that improves accuracy."
    )
    lines.append("")

    lines.append("## Promotion Verdict")
    lines.append("")
    for s in settings:
        if s == CONTROL:
            continue
        lines.append(f"- `{s}`: **{'PROMOTE to held-out validation' if verdicts[s] else 'no-go'}**")
    lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[OK] report written to {out_path}")


if __name__ == "__main__":
    main()
