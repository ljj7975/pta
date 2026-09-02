#!/usr/bin/env python3
"""Analyze floor-ablation sweep output (Exp1) and build 2x2 decomposition table.

Parses three result files:
  - outputs/result_floor_ablation.txt  (FA- labels, this study)
  - outputs/result_repulsion.txt       (REP- labels, Phase 8: confusable-alone)
  - outputs/result_drift_gated_repulsion.txt (DGR- labels, Phase 9: drift+floor)

Produces a 2x2 decomposition table (floor vs. drift), mechanism stats from the
FA record JSONLs, and a verdict on which factor dominates Phase 9's recovery.

Usage:
    python scripts/analyze_floor_ablation.py \
        --out outputs/floor_ablation_report.md
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
FA_LABEL_RE = re.compile(r"^FA-(?P<setting>.+)-s(?P<seed>\d+)$")
REP_LABEL_RE = re.compile(r"^REP-(?P<setting>.+)-s(?P<seed>\d+)$")
DGR_LABEL_RE = re.compile(r"^DGR-(?P<setting>.+)-s(?P<seed>\d+)$")

REGRESSION_BAND = 0.5
GAIN_BAR = 0.3
CONTROL = "control"

# Settings in display order for this study's result file
FA_SETTING_ORDER = [
    "control",
    "lr0.02-f", "lr0.05-f", "lr0.1-f",
    "lr0.02-do", "lr0.05-do",
]

# Phase 8 / 9 dose mapping for 2x2 cells
DOSES = ["0.02", "0.05", "0.1"]


def _parse_result_file(path, label_re):
    """Parse one result file into {setting: {dataset: {seed: acc}}}."""
    out = defaultdict(lambda: defaultdict(dict))
    try:
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
    except FileNotFoundError:
        print(f"WARNING: {path} not found — 2x2 cell will be N/A", file=sys.stderr)
    return out


def _setting_means(parsed):
    """Compute {setting: {dataset: mean_acc}} from parsed results."""
    datasets = sorted({d for ds in parsed.values() for d in ds.keys()})
    return {
        s: {
            d: mean(list(parsed[s].get(d, {}).values()))
            if parsed[s].get(d)
            else None
            for d in datasets
        }
        for s in parsed
    }


def _setting_avg(means, datasets):
    """Average accuracy across datasets for a setting."""
    vals = [means.get(d) for d in datasets]
    return mean([v for v in vals if v is not None])


def mechanism_stats(records_dir, prefix, setting, dataset, seed):
    """Read record JSONL and compute fire-rate / confusability / drift stats."""
    path = Path(records_dir) / f"{prefix}-{setting}-{dataset}-s{seed}" / "records.jsonl"
    if not path.is_file():
        return None
    n_total = 0
    n_applied = 0  # repulsion_applied is True (not False, not None)
    pre_vals, post_vals, drift_vals = [], [], []
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("__header__"):
                continue
            n_total += 1
            # Track drift_velocity for all samples where it's recorded
            if rec.get("drift_velocity") is not None:
                drift_vals.append(rec["drift_velocity"])
            # Fire rate: check repulsion_applied is True (NOT False, NOT None)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--result-fa",
        default="outputs/result_floor_ablation.txt",
        help="Floor ablation result file",
    )
    ap.add_argument(
        "--result-p8",
        default="outputs/result_repulsion.txt",
        help="Phase 8 repulsion result file (confusable-alone)",
    )
    ap.add_argument(
        "--result-p9",
        default="outputs/result_drift_gated_repulsion.txt",
        help="Phase 9 drift-gated repulsion result file",
    )
    ap.add_argument(
        "--records",
        default="outputs/records_floor_ablation",
        help="Record JSONL directory for floor ablation",
    )
    ap.add_argument(
        "--out",
        default="outputs/floor_ablation_report.md",
        help="Output report path",
    )
    args = ap.parse_args()

    # --- Parse all three result files ---
    fa_parsed = _parse_result_file(args.result_fa, FA_LABEL_RE)
    p8_parsed = _parse_result_file(args.result_p8, REP_LABEL_RE)
    p9_parsed = _parse_result_file(args.result_p9, DGR_LABEL_RE)

    fa_means = _setting_means(fa_parsed)
    p8_means = _setting_means(p8_parsed)
    p9_means = _setting_means(p9_parsed)

    # Datasets from the floor ablation study
    datasets = sorted({d for s in fa_parsed.values() for d in s.keys()})
    seeds = sorted({sd for s in fa_parsed.values() for d in s.values() for sd in d.keys()})

    # Control averages (identical across all three files)
    fa_control_avg = _setting_avg(fa_means.get(CONTROL, {}), datasets)
    p8_control_avg = _setting_avg(p8_means.get(CONTROL, {}), datasets)
    p9_control_avg = _setting_avg(p9_means.get(CONTROL, {}), datasets)

    lines = [
        "# Exp1: Floor Ablation — 2×2 Decomposition & Verdict",
        "",
        f"Datasets: {', '.join(datasets)} | Seeds: {seeds}",
        "",
    ]

    # =====================================================================
    # Section 1: Accuracy table from this study's sweep (6 settings)
    # =====================================================================
    lines.append("## 1. Floor Ablation Accuracy (4-seed mean) vs. `control`")
    lines.append("")
    lines.append(
        "| Setting | " + " | ".join(datasets) + " | Avg | Δ vs control |"
    )
    lines.append("|" + "---|" * (len(datasets) + 3))

    fa_deltas = {}
    for s in FA_SETTING_ORDER:
        if s not in fa_means:
            continue
        vals = [fa_means[s].get(d) for d in datasets]
        avg = _setting_avg(fa_means[s], datasets)
        delta = avg - fa_control_avg
        cells = [f"{v:.2f}" if v is not None else "N/A" for v in vals]
        lines.append(
            f"| {s} | " + " | ".join(cells) + f" | {avg:.3f} | {delta:+.3f} |"
        )
        if s != CONTROL:
            fa_deltas[s] = {
                d: (fa_means[s][d] - fa_means[CONTROL][d])
                if (fa_means[s].get(d) is not None and fa_means[CONTROL].get(d) is not None)
                else None
                for d in datasets
            }

    lines.append("")
    lines.append(
        f"Go/no-go bar: avg gain > {GAIN_BAR}pp AND no single-dataset regression > "
        f"{REGRESSION_BAND}pp vs. control."
    )
    lines.append("")

    # =====================================================================
    # Section 2: 2×2 decomposition table (floor × drift)
    # =====================================================================
    lines.append("## 2. 2×2 Decomposition: Floor × Drift")
    lines.append("")
    lines.append(
        "Each cell reports the oxford_pets delta vs. its own-file control at each dose. "
        "The four cells draw from three different result files to isolate each factor."
    )
    lines.append("")

    # Map: (floor, drift) -> (source_label, parsed_means, setting_prefix, source_file)
    # For Phase 8 conf arms: lr0.02-conf, lr0.05-conf, lr0.1-conf
    # For floor-only: lr0.02-f, lr0.05-f, lr0.1-f
    # For drift-only: lr0.02-do, lr0.05-do (no lr0.1-do)
    # For drift+floor: lr0.02-drift, lr0.05-drift, lr0.1-drift

    def _pets_delta(means_dict, control_means, dose):
        """Get oxford_pets delta for a dose from a {setting: means} dict."""
        setting_key = f"lr{dose}"
        m = means_dict.get(setting_key, {})
        pets = m.get("oxford_pets")
        ctrl_pets = control_means.get("oxford_pets")
        if pets is not None and ctrl_pets is not None:
            return pets - ctrl_pets
        return None

    # Build the 2x2 cells
    cells_2x2 = {}  # (floor, dose) -> (setting_key, pets_delta, source_file)

    for dose in DOSES:
        # (floor=none, drift=none) = Phase 8 lr{dose}-conf
        p8_key = f"lr{dose}-conf"
        p8_pets = _pets_delta(
            {dose: p8_means.get(p8_key, {})}, p8_means.get(CONTROL, {}), dose
        )
        # Recompute properly
        if p8_means.get(p8_key, {}).get("oxford_pets") is not None:
            p8_pets = p8_means[p8_key]["oxford_pets"] - p8_means[CONTROL]["oxford_pets"]
        else:
            p8_pets = None

        # (floor=yes, drift=none) = FA lr{dose}-f
        fa_key = f"lr{dose}-f"
        if fa_means.get(fa_key, {}).get("oxford_pets") is not None:
            fa_f_pets = fa_means[fa_key]["oxford_pets"] - fa_means[CONTROL]["oxford_pets"]
        else:
            fa_f_pets = None

        # (floor=none, drift=yes) = FA lr{dose}-do (only 0.02 and 0.05 exist)
        fa_do_key = f"lr{dose}-do"
        if fa_means.get(fa_do_key, {}).get("oxford_pets") is not None:
            fa_do_pets = fa_means[fa_do_key]["oxford_pets"] - fa_means[CONTROL]["oxford_pets"]
        else:
            fa_do_pets = None

        # (floor=yes, drift=yes) = Phase 9 lr{dose}-drift
        p9_key = f"lr{dose}-drift"
        if p9_means.get(p9_key, {}).get("oxford_pets") is not None:
            p9_pets = p9_means[p9_key]["oxford_pets"] - p9_means[CONTROL]["oxford_pets"]
        else:
            p9_pets = None

        cells_2x2[dose] = {
            "none_none": (p8_pets, "result_repulsion.txt (Phase 8)"),
            "yes_none": (fa_f_pets, "result_floor_ablation.txt"),
            "none_yes": (fa_do_pets, "result_floor_ablation.txt"),
            "yes_yes": (p9_pets, "result_drift_gated_repulsion.txt (Phase 9)"),
        }

    # Also compute average deltas across all 3 datasets for each 2x2 cell
    def _avg_delta(p8_m, fa_m, p9_m, ctrl_m, dose, arm_suffix):
        """Compute avg delta across datasets for a 2x2 cell."""
        if arm_suffix == "conf":
            key = f"lr{dose}-conf"
            m, c = p8_m, ctrl_m
        elif arm_suffix == "f":
            key = f"lr{dose}-f"
            m, c = fa_m, ctrl_m
        elif arm_suffix == "do":
            key = f"lr{dose}-do"
            m, c = fa_m, ctrl_m
        elif arm_suffix == "drift":
            key = f"lr{dose}-drift"
            m, c = p9_m, ctrl_m
        else:
            return None
        if key not in m:
            return None
        vals = []
        for d in datasets:
            if m[key].get(d) is not None and c.get(d) is not None:
                vals.append(m[key][d] - c[d])
        return mean(vals) if vals else None

    lines.append("| | drift=none | drift=yes |")
    lines.append("|---|---|---|")
    lines.append("| **floor=none** | (a) confusable-alone | (b) drift-only |")
    lines.append("| **floor=yes** | (c) floor-only | (d) drift+floor |")
    lines.append("")

    # Detailed per-dose 2x2 tables
    lines.append("### Per-dose 2×2 decomposition (oxford_pets Δ)")
    lines.append("")
    lines.append("| Dose | (a) none/none (Phase 8) | (b) none/drift (FA-do) | (c) floor/none (FA-f) | (d) floor/drift (Phase 9) |")
    lines.append("|---|---|---|---|---|")
    for dose in DOSES:
        c = cells_2x2[dose]
        a_str = f"{c['none_none'][0]:+.2f}" if c["none_none"][0] is not None else "N/A"
        b_str = f"{c['none_yes'][0]:+.2f}" if c["none_yes"][0] is not None else "N/A"
        c_str = f"{c['yes_none'][0]:+.2f}" if c["yes_none"][0] is not None else "N/A"
        d_str = f"{c['yes_yes'][0]:+.2f}" if c["yes_yes"][0] is not None else "N/A"
        lines.append(
            f"| {dose} | {a_str} | {b_str} | {c_str} | {d_str} |"
        )
    lines.append("")

    # Source citation
    lines.append("### Source files for each 2×2 cell")
    lines.append("")
    lines.append("| Cell | Setting pattern | Source file |")
    lines.append("|---|---|---|")
    for dose in DOSES:
        c = cells_2x2[dose]
        lines.append(f"| (a) lr{dose} | lr{dose}-conf | {c['none_none'][1]} |")
        lines.append(f"| (b) lr{dose} | lr{dose}-do | {c['none_yes'][1]} |")
        lines.append(f"| (c) lr{dose} | lr{dose}-f | {c['yes_none'][1]} |")
        lines.append(f"| (d) lr{dose} | lr{dose}-drift | {c['yes_yes'][1]} |")
    lines.append("")

    # Cross-check summary: all-dataset avg delta for the 2x2
    lines.append("### Per-dose 2×2 decomposition (avg Δ across all datasets)")
    lines.append("")
    lines.append("| Dose | (a) none/none | (b) none/drift | (c) floor/none | (d) floor/drift |")
    lines.append("|---|---|---|---|---|")
    for dose in DOSES:
        avg_a = _avg_delta(p8_means, fa_means, p9_means, p8_means.get(CONTROL, {}), dose, "conf")
        avg_b = _avg_delta(p8_means, fa_means, p9_means, fa_means.get(CONTROL, {}), dose, "do")
        avg_c = _avg_delta(p8_means, fa_means, p9_means, fa_means.get(CONTROL, {}), dose, "f")
        avg_d = _avg_delta(p8_means, fa_means, p9_means, p9_means.get(CONTROL, {}), dose, "drift")
        a_s = f"{avg_a:+.3f}" if avg_a is not None else "N/A"
        b_s = f"{avg_b:+.3f}" if avg_b is not None else "N/A"
        c_s = f"{avg_c:+.3f}" if avg_c is not None else "N/A"
        d_s = f"{avg_d:+.3f}" if avg_d is not None else "N/A"
        lines.append(f"| {dose} | {a_s} | {b_s} | {c_s} | {d_s} |")
    lines.append("")

    # =====================================================================
    # Section 3: Fire rate + confusability table from record JSONLs
    # =====================================================================
    lines.append("## 3. Mechanism: fire rate, confusability reduction, drift velocity")
    lines.append("")
    lines.append(
        "Pooled across datasets × seeds from `outputs/records_floor_ablation/`. "
        "Control runs record `repulsion_applied=None` (never True) → fire rate = 0."
    )
    lines.append("")
    lines.append(
        "| Setting | Fire rate | Mean pre-confusability | "
        "Mean post-confusability | Mean drift_velocity |"
    )
    lines.append("|---|---|---|---|---|")

    # Control row
    lines.append(f"| control | 0.000 (no-op) | N/A | N/A | N/A |")

    for s in FA_SETTING_ORDER:
        if s == CONTROL:
            continue
        stats_list = []
        for d in datasets:
            for sd in seeds:
                st = mechanism_stats(args.records, "FA", s, d, sd)
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
        lines.append(
            f"| {s} | {fire_rate:.3f} | {pre_str} | {post_str} | {drift_str} |"
        )
    lines.append("")

    # =====================================================================
    # Section 4: Verdict — which factor dominates Phase 9 recovery?
    # =====================================================================
    lines.append("## 4. Verdict: Which factor dominates Phase 9's recovery?")
    lines.append("")
    lines.append(
        "The key question: what drove Phase 9's dramatic improvement on oxford_pets "
        "(-0.86 at lr0.02-conf in Phase 8 → +0.02 at lr0.02-drift in Phase 9)?"
    )
    lines.append("")

    # Build the comparison table across all 4 arms at lr0.02
    lines.append("### oxford_pets Δ at lr=0.02 across all four factor combinations")
    lines.append("")
    lines.append("| Arm | Factor combo | oxford_pets Δ | Source |")
    lines.append("|---|---|---|---|")

    # (a) Phase 8 conf
    p8_pets_02 = None
    if p8_means.get("lr0.02-conf", {}).get("oxford_pets") is not None:
        p8_pets_02 = p8_means["lr0.02-conf"]["oxford_pets"] - p8_means[CONTROL]["oxford_pets"]

    # (b) FA floor-only
    fa_f_pets_02 = None
    if fa_means.get("lr0.02-f", {}).get("oxford_pets") is not None:
        fa_f_pets_02 = fa_means["lr0.02-f"]["oxford_pets"] - fa_means[CONTROL]["oxford_pets"]

    # (c) FA drift-only
    fa_do_pets_02 = None
    if fa_means.get("lr0.02-do", {}).get("oxford_pets") is not None:
        fa_do_pets_02 = fa_means["lr0.02-do"]["oxford_pets"] - fa_means[CONTROL]["oxford_pets"]

    # (d) Phase 9 drift+floor
    p9_pets_02 = None
    if p9_means.get("lr0.02-drift", {}).get("oxford_pets") is not None:
        p9_pets_02 = p9_means["lr0.02-drift"]["oxford_pets"] - p9_means[CONTROL]["oxford_pets"]

    def _fmt(d):
        return f"{d:+.2f}" if d is not None else "N/A"

    lines.append(f"| lr0.02-conf | no floor, no drift | {_fmt(p8_pets_02)} | result_repulsion.txt |")
    lines.append(f"| lr0.02-f | floor only | {_fmt(fa_f_pets_02)} | result_floor_ablation.txt |")
    lines.append(f"| lr0.02-do | drift only | {_fmt(fa_do_pets_02)} | result_floor_ablation.txt |")
    lines.append(f"| lr0.02-drift | floor + drift | {_fmt(p9_pets_02)} | result_drift_gated_repulsion.txt |")
    lines.append("")

    # Also show the same for lr0.05 for completeness
    lines.append("### oxford_pets Δ at lr=0.05 (for cross-reference)")
    lines.append("")
    lines.append("| Arm | Factor combo | oxford_pets Δ | Source |")
    lines.append("|---|---|---|---|")

    for arm, key, factor, src in [
        ("lr0.05-conf", "lr0.05-conf", "no floor, no drift", "result_repulsion.txt"),
        ("lr0.05-f", "lr0.05-f", "floor only", "result_floor_ablation.txt"),
        ("lr0.05-do", "lr0.05-do", "drift only", "result_floor_ablation.txt"),
        ("lr0.05-drift", "lr0.05-drift", "floor + drift", "result_drift_gated_repulsion.txt"),
    ]:
        delta = None
        if arm.startswith("lr"):
            # Use the means from the right file
            for src_means, ctrl_m in [
                (p8_means, p8_means.get(CONTROL, {})),
                (fa_means, fa_means.get(CONTROL, {})),
                (p9_means, p9_means.get(CONTROL, {})),
            ]:
                if src_means.get(key, {}).get("oxford_pets") is not None:
                    delta = src_means[key]["oxford_pets"] - ctrl_m["oxford_pets"]
                    break
        delta_str = f"{delta:+.2f}" if delta is not None else "N/A"
        lines.append(f"| {arm} | {factor} | {delta_str} | {src} |")
    lines.append("")

    # =====================================================================
    # Verdict logic
    # =====================================================================
    # At lr0.02: compare floor-only delta vs no-floor delta (both no-drift)
    # and drift-only delta vs no-drift delta (both no-floor)
    # The dominant factor is whichever closes more of the gap from -0.86 to +0.02

    gap_total = 0.0  # gap to close: from p8_pets_02 to p9_pets_02
    if p8_pets_02 is not None and p9_pets_02 is not None:
        gap_total = p9_pets_02 - p8_pets_02  # positive = improvement

    floor_contribution = 0.0
    drift_contribution = 0.0
    if p8_pets_02 is not None and fa_f_pets_02 is not None:
        floor_contribution = fa_f_pets_02 - p8_pets_02  # improvement from adding floor
    if p8_pets_02 is not None and fa_do_pets_02 is not None:
        drift_contribution = fa_do_pets_02 - p8_pets_02  # improvement from adding drift

    # Determine verdict
    if gap_total == 0:
        verdict = "insufficient data for verdict"
    elif floor_contribution > 0 and abs(floor_contribution) > abs(drift_contribution):
        verdict = (
            f"**Floor dominates.** At lr=0.02 on oxford_pets, adding the floor alone recovers "
            f"{floor_contribution:+.2f}pp of the {gap_total:+.2f}pp gap "
            f"({floor_contribution/gap_total*100:.0f}%), while adding drift alone recovers "
            f"{drift_contribution:+.2f}pp ({drift_contribution/gap_total*100:.0f}%). "
            f"The floor is the primary driver of Phase 9's recovery."
        )
    elif drift_contribution > 0 and abs(drift_contribution) > abs(floor_contribution):
        verdict = (
            f"**Drift dominates.** At lr=0.02 on oxford_pets, adding drift alone recovers "
            f"{drift_contribution:+.2f}pp of the {gap_total:+.2f}pp gap "
            f"({drift_contribution/gap_total*100:.0f}%), while adding the floor alone recovers "
            f"{floor_contribution:+.2f}pp ({floor_contribution/gap_total*100:.0f}%). "
            f"The drift gate is the primary driver of Phase 9's recovery."
        )
    else:
        verdict = (
            f"**Both contribute roughly equally.** At lr=0.02 on oxford_pets, the floor recovers "
            f"{floor_contribution:+.2f}pp and drift recovers {drift_contribution:+.2f}pp of the "
            f"{gap_total:+.2f}pp gap. Neither factor dominates."
        )

    lines.append(f"### Verdict")
    lines.append("")
    lines.append(verdict)
    lines.append("")

    # Exp4 gate assessment
    lines.append("### Exp4 gate assessment")
    lines.append("")
    if p8_pets_02 is not None and fa_do_pets_02 is not None and p9_pets_02 is not None:
        # (drift arms avg Δ) − (floor-only arms avg Δ) at lr=0.02
        gate_a = fa_do_pets_02 - fa_f_pets_02 if fa_f_pets_02 is not None else None
        # drift+floor > floor alone on oxford_pets by >0.1pp
        gate_b = (p9_pets_02 - fa_f_pets_02) if fa_f_pets_02 is not None else None

        gate_a_pass = gate_a is not None and gate_a > 0.1
        gate_b_pass = gate_b is not None and gate_b > 0.1

        lines.append(
            f"- Gate (a): drift-only Δ − floor-only Δ at lr=0.02 = "
            f"{gate_a:+.2f}pp → {'PASS' if gate_a_pass else 'FAIL'} (need >0.1pp)"
        )
        lines.append(
            f"- Gate (b): drift+floor Δ − floor-only Δ on oxford_pets at lr=0.02 = "
            f"{gate_b:+.2f}pp → {'PASS' if gate_b_pass else 'FAIL'} (need >0.1pp)"
        )
        lines.append(
            f"- **Exp4 gate: {'OPEN' if (gate_a_pass or gate_b_pass) else 'CLOSED'}**"
        )
    else:
        lines.append("- Insufficient data for gate assessment")
    lines.append("")

    # =====================================================================
    # Section 5: Go/no-go per setting
    # =====================================================================
    lines.append("## 5. Per-setting go/no-go (standard bar)")
    lines.append("")
    for s in FA_SETTING_ORDER:
        if s == CONTROL:
            continue
        if s not in fa_deltas:
            continue
        avg = _setting_avg(fa_means[s], datasets)
        delta = avg - fa_control_avg
        regressed = any(
            v is not None and v < -REGRESSION_BAND
            for v in fa_deltas[s].values()
        )
        passes = delta > GAIN_BAR and not regressed
        lines.append(
            f"- `{s}`: avg Δ = {delta:+.3f}pp, "
            f"regression > {REGRESSION_BAND}pp: {'YES' if regressed else 'no'} "
            f"→ **{'GO' if passes else 'no-go'}**"
        )
    lines.append("")

    # =====================================================================
    # Write report
    # =====================================================================
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[OK] report written to {out_path}")


if __name__ == "__main__":
    main()
