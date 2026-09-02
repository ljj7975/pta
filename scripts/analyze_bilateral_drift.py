#!/usr/bin/env python3
"""Analyze bilateral-drift experiment (Exp4): OR-condition trigger vs. Phase 9 AND-condition.

Parses outputs/result_bilateral_drift.txt for BD-<setting>-s<seed> labels,
computes 4-seed mean accuracy per (setting, dataset) vs. control, then
compares each bilateral-drift arm against Phase 9's drift_confusable arm
at the same dose -- checking whether the more permissive OR-condition
(drift_ema[top1] OR drift_ema[other] > thresh) improves accuracy.

Also compares fire rates (OR should be strictly more permissive -> higher rate)
and mean pre/post confusability.

Usage:
    python scripts/analyze_bilateral_drift.py \
        --out outputs/bilateral_drift_report.md
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
LABEL_RE = re.compile(r"^BD-(?P<setting>.+)-s(?P<seed>\d+)$")
DGR_LABEL_RE = re.compile(r"^DGR-(?P<setting>.+)-s(?P<seed>\d+)$")

REGRESSION_BAND = 0.5
GAIN_BAR = 0.3
CONTROL = "control"

SETTING_ORDER = ["control", "lr0.02-bd", "lr0.05-bd", "lr0.1-bd"]

# Dose mapping: bilateral_drift setting -> dose label
DOSE_OF_BD = {"lr0.02-bd": "0.02", "lr0.05-bd": "0.05", "lr0.1-bd": "0.1"}
# Phase 9 dose mapping: drift_confusable setting -> dose label
DOSE_OF_DGR = {"lr0.02-drift": "0.02", "lr0.05-drift": "0.05", "lr0.1-drift": "0.1"}

# Fallback Phase 9 values from experimental_results/Phase9_Drift_Gated_Repulsion_Results.md
# lines 35-37 (lr0.02-drift, lr0.05-drift, lr0.1-drift) and lines 64-66 (fire rates).
# Keys: dose -> (avg_accuracy, avg_delta, pets_delta, fire_rate)
PHASE9_FALLBACK = {
    "0.02": {"avg": 70.639, "delta": 0.032, "pets_delta": 0.02, "fire_rate": 0.115},
    "0.05": {"avg": 70.663, "delta": 0.056, "pets_delta": -0.22, "fire_rate": 0.122},
    "0.1":  {"avg": 70.362, "delta": -0.246, "pets_delta": -1.08, "fire_rate": 0.117},
}
PHASE9_FALLBACK_SOURCE = (
    "experimental_results/Phase9_Drift_Gated_Repulsion_Results.md (lines 35-37, 64-66)"
)


def parse_results(path, label_re):
    """Parse a result file into {setting: {dataset: {seed: accuracy}}}."""
    out = defaultdict(lambda: defaultdict(dict))
    with open(path) as fh:
        for line in fh:
            m = RESULT_RE.match(line.strip())
            if not m:
                continue
            lm = label_re.match(m.group("label"))
            if not lm:
                continue
            out[lm.group("setting")][m.group("dataset")][
                int(lm.group("seed"))
            ] = float(m.group("acc"))
    return out


def mechanism_stats(records_dir, setting, dataset, seed):
    """Compute fire rate, mean pre/post confusability for one BD run."""
    path = Path(records_dir) / f"BD-{setting}-{dataset}-s{seed}" / "records.jsonl"
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
            # Use `is True` -- control runs record repulsion_applied=None,
            # firing arms record False/True. Count only True.
            if rec.get("repulsion_applied") is True:
                n_applied += 1
                if rec.get("target_confusability") is not None:
                    pre_vals.append(rec["target_confusability"])
                if rec.get("post_repulsion_confusability") is not None:
                    post_vals.append(rec["post_repulsion_confusability"])
    return {
        "n_total": n_total,
        "n_applied": n_applied,
        "mean_pre": mean(pre_vals) if pre_vals else None,
        "mean_post": mean(post_vals) if post_vals else None,
        "mean_drift": mean(drift_vals) if drift_vals else None,
    }


def dgr_fire_rate(records_dir, dgr_setting, datasets, seeds):
    """Compute pooled fire rate for one Phase 9 DGR drift arm."""
    total_n = total_applied = 0
    for d in datasets:
        for sd in seeds:
            path = Path(records_dir) / f"DGR-{dgr_setting}-{d}-s{sd}" / "records.jsonl"
            if not path.is_file():
                continue
            with open(path) as fh:
                for line in fh:
                    rec = json.loads(line)
                    if rec.get("__header__"):
                        continue
                    total_n += 1
                    if rec.get("repulsion_applied") is True:
                        total_applied += 1
    return total_applied / total_n if total_n else None


def main():
    ap = argparse.ArgumentParser(
        description="Analyze bilateral-drift experiment (Exp4)"
    )
    ap.add_argument(
        "--result",
        default="outputs/result_bilateral_drift.txt",
        help="Path to bilateral drift result file",
    )
    ap.add_argument(
        "--records",
        default="outputs/records_bilateral_drift",
        help="Path to bilateral drift records directory",
    )
    ap.add_argument(
        "--dgr-result",
        default="outputs/result_drift_gated_repulsion.txt",
        help="Path to Phase 9 DGR result file (for matched-dose comparison)",
    )
    ap.add_argument(
        "--dgr-records",
        default="outputs/records_drift_gated_repulsion",
        help="Path to Phase 9 DGR records directory (for fire-rate comparison)",
    )
    ap.add_argument(
        "--out",
        default="outputs/bilateral_drift_report.md",
        help="Output report path",
    )
    args = ap.parse_args()

    # --- Parse bilateral drift results ---
    parsed = parse_results(args.result, LABEL_RE)
    settings = [s for s in SETTING_ORDER if s in parsed] + sorted(
        set(parsed) - set(SETTING_ORDER)
    )
    datasets = sorted({d for ds in parsed.values() for d in ds.keys()})
    seeds = sorted(
        {sd for ds in parsed.values() for d in ds.values() for sd in d.keys()}
    )

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

    # --- Parse Phase 9 DGR results (if available) ---
    dgr_parsed = None
    dgr_available = False
    phase9_source_label = args.dgr_result
    if Path(args.dgr_result).is_file():
        try:
            dgr_parsed = parse_results(args.dgr_result, DGR_LABEL_RE)
            required_drift = {"lr0.02-drift", "lr0.05-drift", "lr0.1-drift"}
            available_drift = required_drift & set(dgr_parsed.keys())
            if len(available_drift) == 3:
                dgr_available = True
            else:
                print(
                    f"[WARN] Phase 9 file found but missing drift settings: "
                    f"{required_drift - available_drift}. Using fallback.",
                    file=sys.stderr,
                )
        except Exception as e:
            print(
                f"[WARN] Could not parse Phase 9 file: {e}. Using fallback.",
                file=sys.stderr,
            )
    else:
        print(
            f"[INFO] Phase 9 file not found -- falling back to report table values "
            f"({PHASE9_FALLBACK_SOURCE})",
            file=sys.stderr,
        )

    # --- Build report lines ---
    lines = [
        "# Exp4: Bilateral-Drift (OR-condition) -- Results",
        "",
        f"Datasets: {', '.join(datasets)} | Seeds: {seeds}",
        "Trigger: bilateral_drift (confusable AND floor AND "
        "(drift_ema[top1] OR drift_ema[other] > thresh))",
        "",
    ]

    # === Section (a): Accuracy table ===
    lines.append(
        "## Overall Accuracy (4-seed mean) vs. `control` (repulsion_lr=0.0)"
    )
    lines.append("")
    lines.append(
        "| Setting | " + " | ".join(datasets) + " | Avg | Δ vs control |"
    )
    lines.append("|" + "---|" * (len(datasets) + 3))

    verdicts = {}
    per_dataset_delta = {}
    for s in settings:
        vals = [means[s].get(d) for d in datasets]
        avg = mean([v for v in vals if v is not None])
        delta = avg - control_avg
        cells = [f"{v:.2f}" if v is not None else "N/A" for v in vals]
        lines.append(
            f"| {s} | " + " | ".join(cells) + f" | {avg:.3f} | {delta:+.3f} |"
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
        f"Go/no-go bar: avg gain > {GAIN_BAR}pp AND "
        f"no single-dataset regression > {REGRESSION_BAND}pp vs. control."
    )
    lines.append("")

    # Per-dataset delta table
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

    # === Section (b): Matched-dose delta table vs Phase 9 ===
    lines.append("## Matched-dose comparison vs. Phase 9 `drift_confusable`")
    lines.append("")
    if dgr_available:
        lines.append(
            f"Phase 9 source: `{phase9_source_label}` (parsed directly). "
            f"Bilateral-drift source: `{args.result}`."
        )
    else:
        lines.append(
            f"Phase 9 fallback: {PHASE9_FALLBACK_SOURCE}. "
            f"Bilateral-drift source: `{args.result}`."
        )
    lines.append("")

    # Compute BD means for matched doses
    bd_means_all = {}
    for s in settings:
        if s == CONTROL:
            continue
        vals = [means[s].get(d) for d in datasets]
        bd_means_all[s] = mean([v for v in vals if v is not None])

    # Compute DGR means for matched doses (if available)
    dgr_means_all = {}
    dgr_control_avg = None
    dgr_control_means = {}
    dgr_pets_delta = {}
    if dgr_available:
        for d in datasets:
            if "control" in dgr_parsed and d in dgr_parsed["control"]:
                dgr_control_means[d] = mean(
                    list(dgr_parsed["control"][d].values())
                )
        if dgr_control_means:
            dgr_control_avg = mean(list(dgr_control_means.values()))
        for dgr_setting in ["lr0.02-drift", "lr0.05-drift", "lr0.1-drift"]:
            if dgr_setting in dgr_parsed:
                dgr_s_means = {}
                for d in datasets:
                    if d in dgr_parsed[dgr_setting]:
                        dgr_s_means[d] = mean(
                            list(dgr_parsed[dgr_setting][d].values())
                        )
                if dgr_s_means:
                    dgr_means_all[dgr_setting] = mean(list(dgr_s_means.values()))
                    if (
                        "oxford_pets" in dgr_s_means
                        and "oxford_pets" in dgr_control_means
                    ):
                        dgr_pets_delta[dgr_setting] = (
                            dgr_s_means["oxford_pets"]
                            - dgr_control_means["oxford_pets"]
                        )

    lines.append(
        "| Dose | BD `bilateral_drift` avg Δ | BD `bilateral_drift` pets Δ | "
        "Phase 9 `drift_confusable` avg Δ | Phase 9 `drift_confusable` pets Δ | "
        "Source (BD / Phase 9) |"
    )
    lines.append("|---|---|---|---|---|---|")

    bd_source = str(args.result)
    for dose in ["0.02", "0.05", "0.1"]:
        bd_setting = f"lr{dose}-bd"
        dgr_setting = f"lr{dose}-drift"

        # BD values
        bd_delta = None
        bd_pets_delta = None
        if bd_setting in per_dataset_delta:
            deltas = per_dataset_delta[bd_setting]
            bd_vals = [v for v in deltas.values() if v is not None]
            if bd_vals:
                bd_delta = mean(bd_vals)
            bd_pets_delta = deltas.get("oxford_pets")

        # Phase 9 values (from parsed file or fallback)
        p9_delta = None
        p9_pets_delta = None
        p9_src = ""
        if dgr_available and dgr_setting in dgr_means_all and dgr_control_avg is not None:
            p9_delta = dgr_means_all[dgr_setting] - dgr_control_avg
            p9_pets_delta = dgr_pets_delta.get(dgr_setting)
            p9_src = phase9_source_label
        elif dose in PHASE9_FALLBACK:
            p9_delta = PHASE9_FALLBACK[dose]["delta"]
            p9_pets_delta = PHASE9_FALLBACK[dose]["pets_delta"]
            p9_src = PHASE9_FALLBACK_SOURCE

        bd_delta_str = f"{bd_delta:+.3f}" if bd_delta is not None else "N/A"
        bd_pets_str = f"{bd_pets_delta:+.2f}" if bd_pets_delta is not None else "N/A"
        p9_delta_str = f"{p9_delta:+.3f}" if p9_delta is not None else "N/A"
        p9_pets_str = f"{p9_pets_delta:+.2f}" if p9_pets_delta is not None else "N/A"
        src_str = f"`{bd_source}` / `{p9_src}`" if p9_src else f"`{bd_source}` / N/A"

        lines.append(
            f"| {dose} | {bd_delta_str} | {bd_pets_str} | "
            f"{p9_delta_str} | {p9_pets_str} | {src_str} |"
        )
    lines.append("")

    # === Section (c): Fire-rate comparison ===
    lines.append("## Fire-rate comparison (pooled across datasets x seeds)")
    lines.append("")
    lines.append(
        "| Setting | BD fire rate | Phase 9 `drift_confusable` fire rate | "
        "OR more permissive? | Source (BD / Phase 9) |"
    )
    lines.append("|---|---|---|---|---|")

    fire_rates_bd = {}
    for s in settings:
        if s == CONTROL:
            continue
        stats_list = []
        for d in datasets:
            for sd in seeds:
                st = mechanism_stats(args.records, s, d, sd)
                if st is not None:
                    stats_list.append(st)
        if stats_list:
            total_n = sum(st["n_total"] for st in stats_list)
            total_applied = sum(st["n_applied"] for st in stats_list)
            fire_rates_bd[s] = total_applied / total_n if total_n else None
        else:
            fire_rates_bd[s] = None

    for dose in ["0.02", "0.05", "0.1"]:
        bd_setting = f"lr{dose}-bd"
        dgr_setting = f"lr{dose}-drift"

        bd_fr = fire_rates_bd.get(bd_setting)

        # Phase 9 fire rate: from records if available, else fallback
        p9_fr = None
        p9_fr_src = ""
        if dgr_available and Path(args.dgr_records).is_dir():
            p9_fr = dgr_fire_rate(
                args.dgr_records, dgr_setting, datasets, seeds
            )
            if p9_fr is not None:
                p9_fr_src = args.dgr_records
        if p9_fr is None and dose in PHASE9_FALLBACK:
            p9_fr = PHASE9_FALLBACK[dose]["fire_rate"]
            p9_fr_src = PHASE9_FALLBACK_SOURCE

        bd_fr_str = f"{bd_fr:.3f}" if bd_fr is not None else "N/A"
        p9_fr_str = f"{p9_fr:.3f}" if p9_fr is not None else "N/A"

        more_permissive = ""
        if bd_fr is not None and p9_fr is not None:
            more_permissive = "YES" if bd_fr >= p9_fr else "NO"
        else:
            more_permissive = "N/A"

        src_cell = f"`{bd_source}` / `{p9_fr_src}`" if p9_fr_src else f"`{bd_source}` / N/A"
        lines.append(
            f"| {dose} | {bd_fr_str} | {p9_fr_str} | "
            f"{more_permissive} | {src_cell} |"
        )
    lines.append("")
    lines.append(
        "The bilateral_drift OR-condition (drift_ema[top1] OR drift_ema[other] > thresh) "
        "is strictly more permissive than Phase 9's AND-condition (drift_ema[top1] > thresh "
        "alone). Fire rates must be >= Phase 9's at matched dose."
    )
    lines.append("")

    # Mean pre/post confusability
    lines.append("### Mean confusability (bilateral_drift arms, pooled across datasets x seeds)")
    lines.append("")
    lines.append("| Setting | Mean pre-confusability | Mean post-confusability | Mean drift_velocity |")
    lines.append("|---|---|---|---|")
    for s in settings:
        if s == CONTROL:
            lines.append(f"| {s} | 0.000 (no-op) | N/A | N/A |")
            continue
        stats_list = []
        for d in datasets:
            for sd in seeds:
                st = mechanism_stats(args.records, s, d, sd)
                if st is not None:
                    stats_list.append(st)
        if not stats_list:
            lines.append(f"| {s} | N/A | N/A | N/A |")
            continue
        pre_vals = [st["mean_pre"] for st in stats_list if st["mean_pre"] is not None]
        post_vals = [st["mean_post"] for st in stats_list if st["mean_post"] is not None]
        drift_vals = [st["mean_drift"] for st in stats_list if st["mean_drift"] is not None]
        pre_str = f"{mean(pre_vals):.4f}" if pre_vals else "N/A"
        post_str = f"{mean(post_vals):.4f}" if post_vals else "N/A"
        drift_str = f"{mean(drift_vals):.4f}" if drift_vals else "N/A"
        lines.append(f"| {s} | {pre_str} | {post_str} | {drift_str} |")
    lines.append("")

    # === Section (d): Verdict ===
    lines.append("## Promotion Verdict")
    lines.append("")
    lines.append(
        f"Standard bar: avg gain > {GAIN_BAR}pp AND no single-dataset regression "
        f"> {REGRESSION_BAND}pp vs. control."
    )
    lines.append("")

    # Per-setting verdict
    for s in settings:
        if s == CONTROL:
            continue
        lines.append(
            f"- `{s}`: **{'PROMOTE to held-out validation' if verdicts[s] else 'no-go'}** "
            f"(avg Δ={bd_means_all[s] - control_avg:+.3f}pp)"
        )
    lines.append("")

    # Cross-study verdict: BD vs Phase 9 drift_confusable at matched dose
    lines.append("### Cross-study verdict: bilateral_drift vs. Phase 9 `drift_confusable`")
    lines.append("")
    improvements = []
    regressions = []
    for dose in ["0.02", "0.05", "0.1"]:
        bd_setting = f"lr{dose}-bd"
        dgr_setting = f"lr{dose}-drift"

        bd_delta = None
        p9_delta = None
        bd_pets = None
        p9_pets = None

        if bd_setting in per_dataset_delta:
            deltas = per_dataset_delta[bd_setting]
            bd_vals = [v for v in deltas.values() if v is not None]
            if bd_vals:
                bd_delta = mean(bd_vals)
            bd_pets = deltas.get("oxford_pets")

        if dgr_available and dgr_setting in dgr_means_all and dgr_control_avg is not None:
            p9_delta = dgr_means_all[dgr_setting] - dgr_control_avg
            p9_pets = dgr_pets_delta.get(dgr_setting)
        elif dose in PHASE9_FALLBACK:
            p9_delta = PHASE9_FALLBACK[dose]["delta"]
            p9_pets = PHASE9_FALLBACK[dose]["pets_delta"]

        if bd_delta is not None and p9_delta is not None:
            avg_diff = bd_delta - p9_delta
            lines.append(
                f"- Dose {dose}: BD avg Δ={bd_delta:+.3f} vs Phase 9 avg Δ={p9_delta:+.3f} "
                f"(diff={avg_diff:+.3f}pp)"
            )
            if avg_diff > 0:
                improvements.append((dose, avg_diff))
            elif avg_diff < 0:
                regressions.append((dose, avg_diff))

            if bd_pets is not None and p9_pets is not None:
                pets_diff = bd_pets - p9_pets
                lines.append(
                    f"  - oxford_pets: BD Δ={bd_pets:+.2f} vs Phase 9 Δ={p9_pets:+.2f} "
                    f"(diff={pets_diff:+.2f}pp)"
                )
        else:
            lines.append(f"- Dose {dose}: incomplete data (BD={bd_delta}, Phase 9={p9_delta})")

    lines.append("")
    # Go/no-go sentence
    both_present = all(
        bd_delta is not None and p9_delta is not None
        for bd_delta, p9_delta in [
            (
                mean([v for v in per_dataset_delta.get(f"lr{d}-bd", {}).values() if v is not None])
                if per_dataset_delta.get(f"lr{d}-bd") else None,
                PHASE9_FALLBACK[d]["delta"]
                if (not dgr_available or f"lr{d}-drift" not in dgr_means_all)
                   and d in PHASE9_FALLBACK
                else (dgr_means_all[f"lr{d}-drift"] - dgr_control_avg
                      if dgr_available and f"lr{d}-drift" in dgr_means_all and dgr_control_avg is not None
                      else None)
            )
            for d in ["0.02", "0.05", "0.1"]
        ]
    )

    if not both_present:
        lines.append(
            "**VERDICT: INCONCLUSIVE** -- insufficient data for cross-study comparison."
        )
    elif improvements and not regressions:
        lines.append(
            f"**VERDICT: DIRECTIONAL GAIN** -- bilateral_drift improves over Phase 9 "
            f"`drift_confusable` at {len(improvements)} of 3 matched doses "
            f"(+{mean([d for _, d in improvements]):.3f}pp avg). "
            f"But no single dose exceeds the +{GAIN_BAR}pp promotion bar on its own "
            f"against control, so this does not constitute a promotion."
        )
    elif improvements and regressions:
        lines.append(
            f"**VERDICT: MIXED** -- bilateral_drift improves at {len(improvements)} dose(s) "
            f"but regresses at {len(regressions)} dose(s) vs. Phase 9 `drift_confusable`. "
            f"No promotion."
        )
    else:
        lines.append(
            f"**VERDICT: NO-GO** -- bilateral_drift does not improve over Phase 9 "
            f"`drift_confusable` at any matched dose. The OR-condition fires more often "
            f"but does not translate to accuracy gains."
        )
    lines.append("")

    # === Write report ===
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[OK] report written to {out_path}")


if __name__ == "__main__":
    main()
