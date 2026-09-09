#!/usr/bin/env python3
"""Analyze scripts/run_drift_gated_repulsion_sweep.sh output (Phase 9).

Parses outputs/result_drift_gated_repulsion.txt for DGR-<setting>-s<seed>
labels, computes 4-seed mean accuracy per (setting, dataset) vs. control,
the mechanism table (fire rate, confusability reduction, drift-velocity
distribution), and the two falsifiable predictions this study exists to
check against Phase 8's confusable-alone numbers
(experimental_results/Phase8_Prototype_Repulsion_Results.md):

  (a) drift_confusable's oxford_pets delta at each matched dose should be
      LESS NEGATIVE than Phase 8's confusable-alone delta at the same dose
      (-0.86 / -3.64 / -8.00 at 0.02 / 0.05 / 0.1).
  (b) stable_confusable's oxford_pets delta should be comparable to or worse
      than those same Phase 8 numbers.

Usage:
    python scripts/analyze_drift_gated_repulsion.py \
        --result outputs/result_drift_gated_repulsion.txt \
        --records outputs/records_drift_gated_repulsion \
        --out outputs/drift_gated_repulsion_report.md
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

RESULT_RE = re.compile(r"^(?P<label>.+)'s performance on (?P<dataset>\S+): Top1- (?P<acc>[\d.]+)\.$")
LABEL_RE = re.compile(r"^DGR-(?P<setting>.+)-s(?P<seed>\d+)$")

REGRESSION_BAND = 0.5
GAIN_BAR = 0.3
CONTROL = "control"

SETTING_ORDER = [
    "control",
    "lr0.02-drift", "lr0.05-drift", "lr0.1-drift",
    "lr0.02-stable", "lr0.05-stable", "lr0.1-stable",
]

# Phase 8 reference: confusable-alone oxford_pets delta vs. its own control, at matched doses.
PHASE8_PETS_DELTA = {"0.02": -0.86, "0.05": -3.64, "0.1": -8.00}
DOSE_OF = {
    "lr0.02-drift": "0.02", "lr0.05-drift": "0.05", "lr0.1-drift": "0.1",
    "lr0.02-stable": "0.02", "lr0.05-stable": "0.05", "lr0.1-stable": "0.1",
}


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
    path = Path(records_dir) / f"DGR-{setting}-{dataset}-s{seed}" / "records.jsonl"
    if not path.is_file():
        return None
    n_total = n_applied = 0
    pre_vals, post_vals, drift_vals = [], [], []
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("__header__"):
                continue
            n_total += 1
            if rec.get("drift_velocity") is not None:
                drift_vals.append(rec["drift_velocity"])
            if rec.get("repulsion_applied"):
                n_applied += 1
                if rec.get("target_confusability") is not None:
                    pre_vals.append(rec["target_confusability"])
                if rec.get("post_repulsion_confusability") is not None:
                    post_vals.append(rec["post_repulsion_confusability"])
    return {
        "n_total": n_total, "n_applied": n_applied,
        "mean_pre": mean(pre_vals) if pre_vals else None,
        "mean_post": mean(post_vals) if post_vals else None,
        "mean_drift": mean(drift_vals) if drift_vals else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default="outputs/result_drift_gated_repulsion.txt")
    ap.add_argument("--records", default="outputs/records_drift_gated_repulsion")
    ap.add_argument("--out", default="outputs/drift_gated_repulsion_report.md")
    args = ap.parse_args()

    parsed = parse_results(args.result)
    settings = [s for s in SETTING_ORDER if s in parsed] + sorted(set(parsed) - set(SETTING_ORDER))
    datasets = sorted({d for ds in parsed.values() for d in ds.keys()})
    seeds = sorted({sd for ds in parsed.values() for d in ds.values() for sd in d.keys()})

    means = {s: {d: mean(list(parsed[s].get(d, {}).values())) if parsed[s].get(d) else None for d in datasets} for s in settings}
    control_means = means.get(CONTROL, {})
    control_avg = mean([v for v in control_means.values() if v is not None])

    lines = ["# Phase 9: Drift-Gated Prototype Repulsion — Results", ""]
    lines.append(f"Datasets: {', '.join(datasets)} | Seeds: {seeds}")
    lines.append("")
    lines.append("## Overall Accuracy (4-seed mean) vs. `control` (repulsion_lr=0.0)")
    lines.append("")
    lines.append("| Setting | " + " | ".join(datasets) + " | Avg | Δ vs control |")
    lines.append("|" + "---|" * (len(datasets) + 3))

    verdicts = {}
    per_dataset_delta = {}
    for s in settings:
        vals = [means[s].get(d) for d in datasets]
        avg = mean([v for v in vals if v is not None])
        delta = avg - control_avg
        cells = [f"{v:.2f}" if v is not None else "N/A" for v in vals]
        lines.append(f"| {s} | " + " | ".join(cells) + f" | {avg:.3f} | {delta:+.3f} |")
        if s != CONTROL:
            per_dataset_delta[s] = {
                d: (means[s][d] - control_means[d]) if (means[s].get(d) is not None and control_means.get(d) is not None) else None
                for d in datasets
            }
            regressed = any(v is not None and v < -REGRESSION_BAND for v in per_dataset_delta[s].values())
            verdicts[s] = (delta > GAIN_BAR and not regressed)

    lines.append("")
    lines.append(f"Go/no-go bar: avg gain > {GAIN_BAR}pp AND no single-dataset regression > {REGRESSION_BAND}pp vs. control.")
    lines.append("")

    lines.append("## Per-dataset Δ (accuracy points vs. control)")
    lines.append("")
    lines.append("| Setting | " + " | ".join(f"{d} Δ" for d in datasets) + " |")
    lines.append("|" + "---|" * (len(datasets) + 1))
    for s in settings:
        if s == CONTROL:
            continue
        row = [f"{per_dataset_delta[s][d]:+.2f}" if per_dataset_delta[s].get(d) is not None else "N/A" for d in datasets]
        lines.append(f"| {s} | " + " | ".join(row) + " |")
    lines.append("")

    # --- Mechanism table ---
    lines.append("## Mechanism: fire rate, confusability reduction, drift velocity (pooled across datasets x seeds)")
    lines.append("")
    lines.append("| Setting | Fire rate | Mean pre-confusability | Mean post-confusability | Mean drift_velocity |")
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
        drift_vals = [st["mean_drift"] for st in stats_list if st["mean_drift"] is not None]
        pre_str = f"{mean(pre_vals):.4f}" if pre_vals else "N/A"
        post_str = f"{mean(post_vals):.4f}" if post_vals else "N/A"
        drift_str = f"{mean(drift_vals):.4f}" if drift_vals else "N/A"
        lines.append(f"| {s} | {fire_rate:.3f} | {pre_str} | {post_str} | {drift_str} |")
    lines.append("")

    # --- Falsifiable predictions vs. Phase 8 ---
    lines.append("## Falsifiable predictions vs. Phase 8's `confusable`-alone oxford_pets deltas")
    lines.append("")
    lines.append("| Dose | Phase 8 `confusable` (no drift gate) | `drift_confusable` (this study) | `stable_confusable` (this study) | Prediction holds? |")
    lines.append("|---|---|---|---|---|")
    for dose in ["0.02", "0.05", "0.1"]:
        drift_key = f"lr{dose}-drift"
        stable_key = f"lr{dose}-stable"
        p8 = PHASE8_PETS_DELTA[dose]
        drift_delta = per_dataset_delta.get(drift_key, {}).get("oxford_pets")
        stable_delta = per_dataset_delta.get(stable_key, {}).get("oxford_pets")
        drift_str = f"{drift_delta:+.2f}" if drift_delta is not None else "N/A"
        stable_str = f"{stable_delta:+.2f}" if stable_delta is not None else "N/A"
        holds = "N/A"
        if drift_delta is not None and stable_delta is not None:
            pred_a = drift_delta > p8  # less negative
            pred_b = stable_delta <= p8 + 0.5  # comparable or worse (allow 0.5pp slack)
            holds = f"a: {'YES' if pred_a else 'NO'}, b: {'YES' if pred_b else 'NO'}"
        lines.append(f"| {dose} | {p8:+.2f} | {drift_str} | {stable_str} | {holds} |")
    lines.append("")
    lines.append(
        "Prediction (a): drift_confusable's oxford_pets delta should be less negative than Phase 8's "
        "confusable-alone delta at the same dose. Prediction (b): stable_confusable's delta should be "
        "comparable to or worse than Phase 8's. Note both new arms also include the repulsion floor "
        "(absent in Phase 8), so any difference from Phase 8 is a combined-effect comparison; the "
        "drift_confusable vs. stable_confusable comparison alone isolates the drift-gating variable."
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
