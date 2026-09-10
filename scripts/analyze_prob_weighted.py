#!/usr/bin/env python3
"""Analyze probability-weighted EMA sweep output (Exp3, Phase 10).

Parses outputs/result_prob_weighted.txt for PW-<setting>-s<seed> labels,
computes 4-seed mean accuracy per (setting, dataset) vs. control (gamma=0),
monotonicity across gamma values, convergence speed, and a go/no-go verdict.

Usage:
    python scripts/analyze_prob_weighted.py \
        --result outputs/result_prob_weighted.txt \
        --records outputs/records_prob_weighted \
        --out outputs/prob_weighted_report.md
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

RESULT_RE = re.compile(
    r"^(?P<label>.+)'s performance on (?P<dataset>\S+): Top1- (?P<acc>[\d.]+)\.$"
)
LABEL_RE = re.compile(r"^PW-(?P<setting>.+)-s(?P<seed>\d+)$")

REGRESSION_BAND = 0.5
GAIN_BAR = 0.3
CONTROL = "control"

# gamma value for each setting (used for monotonicity check)
GAMMA_OF = {"control": 0.0, "g0.5": 0.5, "g1.0": 1.0, "g2.0": 2.0}

# Canonical top-1 EMA write reference: dtd / oxford_flowers / oxford_pets (±0.10pp)
REFERENCE_CONTROL = {"dtd": 46.85, "oxford_flowers": 74.22, "oxford_pets": 90.75}
REF_TOLERANCE = 0.10

# Ordered by gamma for monotonicity check
SETTING_ORDER = ["control", "g0.5", "g1.0", "g2.0"]


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
            out[lm.group("setting")][m.group("dataset")][int(lm.group("seed"))] = float(
                m.group("acc")
            )
    return out


def load_convergence(records_dir, setting, dataset, seed):
    """Load convergence.json from a record dir; return dict or None."""
    path = Path(records_dir) / f"PW-{setting}-{dataset}-s{seed}" / "convergence.json"
    if not path.is_file():
        return None
    with open(path) as fh:
        return json.loads(fh.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--result", default="outputs/result_prob_weighted.txt"
    )
    ap.add_argument(
        "--records", default="outputs/records_prob_weighted"
    )
    ap.add_argument(
        "--out", default="outputs/prob_weighted_report.md"
    )
    args = ap.parse_args()

    parsed = parse_results(args.result)
    settings = [s for s in SETTING_ORDER if s in parsed] + sorted(
        set(parsed) - set(SETTING_ORDER)
    )
    datasets = sorted({d for ds in parsed.values() for d in ds.keys()})
    seeds = sorted(
        {sd for ds in parsed.values() for d in ds.values() for sd in d.keys()}
    )

    # Compute 4-seed means
    means = {
        s: {
            d: mean(list(parsed[s].get(d, {}).values()))
            if parsed[s].get(d)
            else None
            for d in datasets
        }
        for s in settings
    }

    # --- Validate control identity ---
    control_means = means.get(CONTROL, {})
    lines = []
    validated = True

    lines.append("# Phase 10, Exp3: Probability-Weighted EMA — Results")
    lines.append("")
    lines.append(f"Datasets: {', '.join(datasets)} | Seeds: {seeds}")
    lines.append("")

    lines.append("## Control Identity Check")
    lines.append("")
    lines.append(
        "gamma=0.0 is the canonical top-1 write; control must reproduce the "
        "published reference 46.85 / 74.22 / 90.75 (±0.10pp)."
    )
    lines.append("")
    lines.append("| Dataset | Reference | Control mean | Match? |")
    lines.append("|---|---|---|---|")

    for ds, ref_val in REFERENCE_CONTROL.items():
        ctrl_val = control_means.get(ds)
        if ctrl_val is None:
            match_str = "MISSING"
            validated = False
        else:
            delta = abs(ctrl_val - ref_val)
            ok = delta <= REF_TOLERANCE
            match_str = f"{'YES' if ok else 'NO — MISMATCH ({delta:.3f}pp)'}"
            if not ok:
                validated = False
        ctrl_str = f"{ctrl_val:.2f}" if ctrl_val is not None else "N/A"
        lines.append(f"| {ds} | {ref_val:.2f} | {ctrl_str} | {match_str} |")
    lines.append("")

    if not validated:
        lines.append(
            "**WARNING: Control identity check FAILED. The gamma=0 control row "
            "does not match the published reference. This means something other "
            "than p^gamma was changed. DO NOT trust any other row — debug before "
            "interpreting results.**"
        )
        lines.append("")
    else:
        lines.append("Control identity: **PASS** — all datasets within tolerance.")
        lines.append("")

    control_avg = mean([v for v in control_means.values() if v is not None])

    # --- Overall accuracy table ---
    lines.append(
        "## Overall Accuracy (4-seed mean) vs. `control` (gamma=0)"
    )
    lines.append("")
    lines.append(
        "| Setting (gamma) | "
        + " | ".join(datasets)
        + " | Avg | Δ vs control |"
    )
    lines.append("|" + "---|" * (len(datasets) + 3))

    verdicts = {}
    per_dataset_delta = {}
    setting_avgs = {}  # setting -> avg accuracy

    for s in settings:
        vals = [means[s].get(d) for d in datasets]
        avg = mean([v for v in vals if v is not None])
        delta = avg - control_avg
        gamma_str = f"{GAMMA_OF[s]:.1f}" if s in GAMMA_OF else "?"
        cells = [f"{v:.2f}" if v is not None else "N/A" for v in vals]
        lines.append(
            f"| {s} (γ={gamma_str}) | "
            + " | ".join(cells)
            + f" | {avg:.3f} | {delta:+.3f} |"
        )
        setting_avgs[s] = avg

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
        f"Go/no-go bar: avg gain > {GAIN_BAR}pp AND no single-dataset regression "
        f"> {REGRESSION_BAND}pp vs. control."
    )
    lines.append("")

    # --- Per-dataset delta table ---
    lines.append("## Per-dataset Δ (accuracy points vs. control)")
    lines.append("")
    lines.append(
        "| Setting | "
        + " | ".join(f"{d} Δ" for d in datasets)
        + " |"
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

    # --- Monotonicity check ---
    lines.append("## Monotonicity Across Gamma")
    lines.append("")
    lines.append(
        "Does accuracy degrade monotonically as gamma increases? "
        "(p^gamma downweights low-confidence top-1 writes; gamma>0 should "
        "hurt harder datasets more.)"
    )
    lines.append("")

    # Build gamma-ordered list of settings that exist
    gamma_settings = sorted(
        [(GAMMA_OF[s], s) for s in settings if s in GAMMA_OF], key=lambda x: x[0]
    )

    # Check per-dataset monotonicity
    mono_report = []
    overall_mono = True
    for ds in datasets:
        vals_at_gamma = []
        for g, s in gamma_settings:
            v = means[s].get(ds)
            if v is not None:
                vals_at_gamma.append((g, v))
        # Check non-increasing (degradation = values decrease or stay same as gamma increases)
        ds_mono = True
        for i in range(1, len(vals_at_gamma)):
            if vals_at_gamma[i][1] > vals_at_gamma[i - 1][1] + 0.01:
                # Not strictly monotonic — accuracy increased with gamma
                ds_mono = False
                overall_mono = False
        direction = "decreasing" if ds_mono else "NOT monotonic"
        mono_report.append((ds, direction, vals_at_gamma))

    for ds, direction, vals in mono_report:
        val_str = " → ".join(f"{v:.2f}" for _, v in vals)
        lines.append(f"- **{ds}** ({direction}): {val_str}")

    lines.append("")
    if overall_mono:
        lines.append(
            "**Monotonicity: YES** — accuracy decreases monotonically with increasing "
            "gamma on all datasets, as predicted by the p^gamma downweighting mechanism."
        )
    else:
        lines.append(
            "**Monotonicity: PARTIAL** — accuracy does not decrease monotonically on "
            "all datasets. Some gamma>0 values show slight gains over lower gamma."
        )
    lines.append("")

    # --- Convergence table ---
    lines.append(
        "## Convergence (pooled mean across 3 datasets × 4 seeds)"
    )
    lines.append("")
    lines.append("| Setting (gamma) | acc@10% | acc@25% | acc@50% | acc@75% | final | auc_norm |")
    lines.append("|---|---|---|---|---|---|---|")

    for s in settings:
        acc_10, acc_25, acc_50, acc_75, final_acc, auc_vals = [], [], [], [], [], []
        for d in datasets:
            for sd in seeds:
                conv = load_convergence(args.records, s, d, sd)
                if conv is None:
                    continue
                af = conv.get("acc_at_fractions", {})
                if "0.1" in af:
                    acc_10.append(af["0.1"])
                if "0.25" in af:
                    acc_25.append(af["0.25"])
                if "0.5" in af:
                    acc_50.append(af["0.5"])
                if "0.75" in af:
                    acc_75.append(af["0.75"])
                if "final_acc" in conv:
                    final_acc.append(conv["final_acc"])
                if "auc_norm" in conv:
                    auc_vals.append(conv["auc_norm"])

        def pooled_mean(lst):
            return f"{mean(lst):.2f}" if lst else "N/A"

        gamma_str = f"{GAMMA_OF[s]:.1f}" if s in GAMMA_OF else "?"
        lines.append(
            f"| {s} (γ={gamma_str}) | "
            + " | ".join(
                [
                    pooled_mean(acc_10),
                    pooled_mean(acc_25),
                    pooled_mean(acc_50),
                    pooled_mean(acc_75),
                    pooled_mean(final_acc),
                    pooled_mean(auc_vals),
                ]
            )
            + " |"
        )
    lines.append("")

    # --- Go/No-Go verdict ---
    lines.append("## Go/No-Go Verdict")
    lines.append("")

    any_pass = False
    for s in settings:
        if s == CONTROL:
            continue
        delta = setting_avgs[s] - control_avg
        gamma_str = f"{GAMMA_OF[s]:.1f}" if s in GAMMA_OF else "?"
        status = "PASS" if verdicts.get(s, False) else "no-go"
        if verdicts.get(s, False):
            any_pass = True

        # Detail the reason
        if not verdicts.get(s, False):
            if delta <= GAIN_BAR:
                reason = f"avg delta {delta:+.3f}pp ≤ {GAIN_BAR}pp gain bar"
            else:
                regressed_ds = [
                    d
                    for d in datasets
                    if per_dataset_delta[s].get(d) is not None
                    and per_dataset_delta[s][d] < -REGRESSION_BAND
                ]
                reason = f"regression > {REGRESSION_BAND}pp on: {', '.join(regressed_ds)}"
        else:
            reason = f"avg delta {delta:+.3f}pp > {GAIN_BAR}pp, no regression > {REGRESSION_BAND}pp"

        lines.append(f"- `{s}` (γ={gamma_str}): **{status}** — {reason}")

    lines.append("")

    # One explicit go/no-go sentence
    if any_pass:
        lines.append(
            "**OVERALL: GO** — at least one gamma>0 setting improves avg accuracy "
            f"(> {GAIN_BAR}pp gain) without single-dataset regression > "
            f"{REGRESSION_BAND}pp."
        )
    else:
        lines.append(
            "**OVERALL: NO-GO** — no gamma>0 setting improves avg accuracy above "
            f"the {GAIN_BAR}pp gain bar without single-dataset regression > "
            f"{REGRESSION_BAND}pp. Probability-weighted EMA (p^gamma downweighting "
            "of low-confidence top-1 writes) does not improve over the canonical "
            "uniform top-1 write."
        )
    lines.append("")

    # --- Write report ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[OK] report written to {out_path}")


if __name__ == "__main__":
    main()
