#!/usr/bin/env python3
"""Analyze text-anchored EMA sweep output (Phase 10, Exp2).

Parses outputs/result_text_anchored.txt for TA-<setting>-s<seed> labels,
computes 4-seed mean accuracy per (setting, dataset) vs. control, reports
monotonicity across anchor_mix, convergence stats from the record dirs'
convergence.json, and a go/no-go verdict.

Usage:
    python scripts/analyze_text_anchored.py \
        --result outputs/result_text_anchored.txt \
        --records outputs/records_text_anchored \
        --out outputs/text_anchored_report.md
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

RESULT_RE = re.compile(
    r"^(?P<label>.+)'s performance on (?P<dataset>\S+): Top1- (?P<acc>[\d.]+)\.$"
)
LABEL_RE = re.compile(r"^TA-(?P<setting>.+)-s(?P<seed>\d+)$")

REGRESSION_BAND = 0.5
GAIN_BAR = 0.3
CONTROL = "control"

# anchor_mix values descend from 1.0 (full image) toward 0.8 (more text).
SETTING_ORDER = ["control", "a0.99", "a0.95", "a0.90", "a0.80"]

# anchor_mix value implied by each setting label
ANCHOR_MIX_OF = {
    "control": 1.0,
    "a0.99": 0.99,
    "a0.95": 0.95,
    "a0.90": 0.90,
    "a0.80": 0.80,
}

# Convergence percentile thresholds to report
CONVERGENCE_FRACTIONS = ["0.1", "0.25", "0.5", "0.75", "1.0"]
CONVERGENCE_LABELS = ["acc@10%", "acc@25%", "acc@50%", "acc@75%", "final"]


def parse_results(path):
    """Parse result file into nested dict: setting -> dataset -> seed -> acc."""
    out = defaultdict(lambda: defaultdict(dict))
    with open(path) as fh:
        for line in fh:
            m = RESULT_RE.match(line.strip())
            if not m:
                continue
            lm = LABEL_RE.match(m.group("label"))
            if not lm:
                continue
            out[lm.group("setting")][m.group("dataset")][int(lm.group("seed"))] = (
                float(m.group("acc"))
            )
    return out


def load_convergence(records_dir, setting, dataset, seed):
    """Load convergence.json from a record subdir, return dict or None."""
    path = Path(records_dir) / f"TA-{setting}-{dataset}-s{seed}" / "convergence.json"
    if not path.is_file():
        return None
    with open(path) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result", default="outputs/result_text_anchored.txt")
    ap.add_argument("--records", default="outputs/records_text_anchored")
    ap.add_argument("--out", default="outputs/text_anchored_report.md")
    args = ap.parse_args()

    parsed = parse_results(args.result)
    settings = [s for s in SETTING_ORDER if s in parsed] + sorted(
        set(parsed) - set(SETTING_ORDER)
    )
    datasets = sorted({d for ds in parsed.values() for d in ds.keys()})
    seeds = sorted({sd for ds in parsed.values() for d in ds.values() for sd in d.keys()})

    # 4-seed mean per (setting, dataset)
    means = {
        s: {
            d: mean(list(parsed[s].get(d, {}).values()))
            if parsed[s].get(d)
            else None
            for d in datasets
        }
        for s in settings
    }
    control_means = means.get(CONTROL, {})
    control_avg = mean([v for v in control_means.values() if v is not None])

    # ---- Report ----
    lines = ["# Phase 10: Text-Anchored EMA — Results", ""]
    lines.append(f"Datasets: {', '.join(datasets)} | Seeds: {seeds}")
    lines.append("")
    lines.append(
        "## Overall Accuracy (4-seed mean) vs. `control` (anchor_mix=1.0)"
    )
    lines.append("")
    lines.append(
        "| Setting | anchor_mix | "
        + " | ".join(datasets)
        + " | Avg | Δ vs control |"
    )
    lines.append("|" + "---|" * (len(datasets) + 4))

    verdicts = {}
    per_dataset_delta = {}
    for s in settings:
        vals = [means[s].get(d) for d in datasets]
        avg = mean([v for v in vals if v is not None])
        delta = avg - control_avg
        cells = [f"{v:.2f}" if v is not None else "N/A" for v in vals]
        amix = ANCHOR_MIX_OF.get(s, "N/A")
        lines.append(
            f"| {s} | {amix} | "
            + " | ".join(cells)
            + f" | {avg:.3f} | {delta:+.3f} |"
        )
        if s != CONTROL:
            per_dataset_delta[s] = {
                d: (means[s][d] - control_means[d])
                if (means[s].get(d) is not None and control_means.get(d) is not None)
                else None
                for d in datasets
            }
            regressed = any(
                v is not None and v < -REGRESSION_BAND
                for v in per_dataset_delta[s].values()
            )
            verdicts[s] = delta > GAIN_BAR and not regressed

    lines.append("")
    lines.append(
        f"Go/no-go bar: avg gain > {GAIN_BAR}pp AND no single-dataset "
        f"regression > {REGRESSION_BAND}pp vs. control."
    )
    lines.append("")

    # ---- Per-dataset delta table ----
    lines.append("## Per-dataset Δ (accuracy points vs. control)")
    lines.append("")
    lines.append(
        "| Setting | " + " | ".join(f"{d} Δ" for d in datasets) + " |"
    )
    lines.append("|" + "---|" * (len(datasets) + 1))
    for s in settings:
        if s == CONTROL:
            continue
        row = [
            f"{per_dataset_delta[s][d]:+.2f}"
            if per_dataset_delta[s].get(d) is not None
            else "N/A"
            for d in datasets
        ]
        lines.append(f"| {s} | " + " | ".join(row) + " |")
    lines.append("")

    # ---- Monotonicity across anchor_mix ----
    lines.append("## Monotonicity across anchor_mix (1.0 → 0.8)")
    lines.append("")
    # For each dataset, check whether avg accuracy is monotonic as anchor_mix decreases
    mono_results = {}
    for d in datasets:
        accs = []
        for s in settings:
            v = means[s].get(d)
            if v is not None:
                accs.append((ANCHOR_MIX_OF.get(s, 0), v))
        # Sort by anchor_mix descending (1.0 first)
        accs.sort(key=lambda x: -x[0])
        vals_only = [a for _, a in accs]
        improving = all(
            vals_only[i] <= vals_only[i + 1] for i in range(len(vals_only) - 1)
        )
        degrading = all(
            vals_only[i] >= vals_only[i + 1] for i in range(len(vals_only) - 1)
        )
        if improving:
            mono_results[d] = "improving (accuracy rises as text anchoring strengthens)"
        elif degrading:
            mono_results[d] = "degrading (accuracy falls as text anchoring strengthens)"
        else:
            mono_results[d] = "non-monotonic"
        lines.append(f"- **{d}**: {mono_results[d]}")

    overall_mono_vals = []
    for s in settings:
        v = means[s].get("Avg")
    # Overall avg trend
    avg_by_setting = []
    for s in settings:
        vals = [means[s].get(d) for d in datasets]
        avg_by_setting.append(
            (ANCHOR_MIX_OF.get(s, 0), mean([v for v in vals if v is not None]))
        )
    avg_by_setting.sort(key=lambda x: -x[0])
    avg_vals = [a for _, a in avg_by_setting]
    if all(avg_vals[i] <= avg_vals[i + 1] for i in range(len(avg_vals) - 1)):
        overall = "improving overall"
    elif all(avg_vals[i] >= avg_vals[i + 1] for i in range(len(avg_vals) - 1)):
        overall = "degrading overall"
    else:
        overall = "non-monotonic overall"
    lines.append(f"- **Overall**: {overall}")
    lines.append("")

    # ---- Convergence table ----
    lines.append(
        "## Convergence (pooled mean across 3 datasets × 4 seeds)"
    )
    lines.append("")
    lines.append(
        "| Setting | "
        + " | ".join(CONVERGENCE_LABELS + ["auc_norm"])
        + " |"
    )
    lines.append("|" + "---|" * (len(CONVERGENCE_LABELS) + 2))

    conv_means = {}
    for s in settings:
        all_convs = []
        for d in datasets:
            for sd in seeds:
                cv = load_convergence(args.records, s, d, sd)
                if cv is not None:
                    all_convs.append(cv)
        if not all_convs:
            conv_means[s] = None
            rows = ["N/A"] * (len(CONVERGENCE_LABELS) + 1)
            lines.append(f"| {s} | " + " | ".join(rows) + " |")
            continue

        # Extract acc_at_fractions for each threshold
        frac_vals = defaultdict(list)
        auc_vals = []
        for cv in all_convs:
            af = cv.get("acc_at_fractions", {})
            for frac in CONVERGENCE_FRACTIONS:
                if frac in af:
                    frac_vals[frac].append(af[frac])
            if "auc_norm" in cv:
                auc_vals.append(cv["auc_norm"])

        cells = []
        for frac in CONVERGENCE_FRACTIONS:
            vals = frac_vals.get(frac, [])
            cells.append(f"{mean(vals):.2f}" if vals else "N/A")
        auc_str = f"{mean(auc_vals):.4f}" if auc_vals else "N/A"
        cells.append(auc_str)
        conv_means[s] = {
            frac: mean(frac_vals.get(frac, [])) for frac in CONVERGENCE_FRACTIONS
        }
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- Control identity check ----
    lines.append("## Control identity check (vs. published reference)")
    lines.append("")
    ref = {"dtd": 46.85, "oxford_flowers": 74.22, "oxford_pets": 90.75}
    for d in ["dtd", "oxford_flowers", "oxford_pets"]:
        cv = control_means.get(d)
        rv = ref.get(d)
        if cv is not None and rv is not None:
            diff = abs(cv - rv)
            ok = "PASS" if diff <= 0.10 else "FAIL"
            lines.append(f"- {d}: computed {cv:.2f}, reference {rv:.2f}, |Δ|={diff:.2f}pp — {ok}")
    lines.append("")

    # ---- Promotion verdict ----
    lines.append("## Go/No-Go Verdict")
    lines.append("")
    any_go = False
    for s in settings:
        if s == CONTROL:
            continue
        decision = "PROMOTE to held-out validation" if verdicts[s] else "no-go"
        if verdicts[s]:
            any_go = True
        lines.append(f"- `{s}` (anchor_mix={ANCHOR_MIX_OF.get(s, '?')}): **{decision}**")
    lines.append("")
    if any_go:
        lines.append(
            "**VERDICT: GO** — at least one anchor_mix < 1.0 beats control by > "
            f"{GAIN_BAR}pp avg with no dataset regression > {REGRESSION_BAND}pp."
        )
    else:
        lines.append(
            "**VERDICT: NO-GO** — no anchor_mix < 1.0 achieves > "
            f"{GAIN_BAR}pp avg gain over control without a dataset regression > "
            f"{REGRESSION_BAND}pp."
        )
    lines.append("")
    lines.append(
        "Note: This verdict is for text-anchored EMA in isolation. Combining "
        "with other mechanisms (e.g., prototype repulsion) is outside the scope "
        "of this analysis."
    )
    lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[OK] report written to {out_path}")


if __name__ == "__main__":
    main()
