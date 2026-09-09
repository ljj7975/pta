#!/usr/bin/env python3
"""Phase 1 analysis: trust-adaptive read-time fusion sweep.

Reads:
  - outputs/result_trust_fusion.txt   (overall top-1 accuracy per run)
  - outputs/records_trust_fusion/TFP-<setting>-<dataset>-s<seed>/records.jsonl
                                       (per-sample records with tau_eff/trust_regime)

Reports (mirrors experimental_results/*.md methodology):
  1. Overall accuracy per (setting, dataset), 4-seed mean vs. the `control`
     setting ((tau_scale_trusted, tau_scale_untrusted) == (1.0, 1.0)).
  2. Tie-only accuracy per setting (clip.argmax != image_proto.argmax) — the
     metric that most directly tests whether trust-adaptive fusion improves
     tie-breaking (PTA_Limitations_and_Patch_Signal_Analysis.md Part 1.1).
  3. Accuracy by trust_regime (trusted/untrusted) to sanity-check the
     mechanism is behaving as designed.
  4. A go/no-go verdict per setting against the plan's numeric bar: >0.3pp
     avg gain over control, no dataset regression beyond ~0.5pp.

Usage:
    python scripts/analyze_trust_fusion.py \
        --result outputs/result_trust_fusion.txt \
        --records outputs/records_trust_fusion \
        --out outputs/trust_fusion_report.md
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

RESULT_RE = re.compile(r"^(?P<label>.+)'s performance on (?P<dataset>\S+): Top1- (?P<acc>[\d.]+)\.$")
LABEL_RE = re.compile(r"^TFP-(?P<setting>.+)-s(?P<seed>\d+)$")

REGRESSION_BAND = 0.5   # pp; per-dataset noise band (PatchModPTA_..._Trust_Analysis.md 4b.3)
GAIN_BAR = 0.3           # pp; avg-over-datasets promotion bar
CONTROL = "control"


def parse_results(path):
    """-> {setting: {dataset: {seed: acc}}}"""
    out = defaultdict(lambda: defaultdict(dict))
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            m = RESULT_RE.match(line)
            if not m:
                continue
            lm = LABEL_RE.match(m.group("label"))
            if not lm:
                continue
            setting = lm.group("setting")
            seed = int(lm.group("seed"))
            dataset = m.group("dataset")
            out[setting][dataset][seed] = float(m.group("acc"))
    return out


def argmax(vec):
    if not vec:
        return None
    return max(range(len(vec)), key=lambda i: vec[i])


def load_records(records_dir, setting, dataset, seed):
    label = f"TFP-{setting}-{dataset}-s{seed}"
    path = Path(records_dir) / label / "records.jsonl"
    samples = []
    if not path.is_file():
        return samples
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("__header__"):
                continue
            samples.append(rec)
    return samples


def tie_and_quadrant_stats(records_dir, settings, datasets, seeds):
    """Per setting: tie-only accuracy (4-seed pooled) and per-regime accuracy."""
    stats = {}
    for setting in settings:
        tie_correct = tie_total = 0
        regime_correct = defaultdict(int)
        regime_total = defaultdict(int)
        for dataset in datasets:
            for seed in seeds:
                for s in load_records(records_dir, setting, dataset, seed):
                    lg = s.get("logits") or {}
                    clip_a = argmax(lg.get("clip"))
                    proto_a = argmax(lg.get("image_proto"))
                    if clip_a is not None and proto_a is not None and clip_a != proto_a:
                        tie_total += 1
                        if s.get("correct"):
                            tie_correct += 1
                    regime = s.get("trust_regime")
                    if regime:
                        regime_total[regime] += 1
                        if s.get("correct"):
                            regime_correct[regime] += 1
        stats[setting] = {
            "tie_acc": (100.0 * tie_correct / tie_total) if tie_total else None,
            "tie_total": tie_total,
            "regime_acc": {
                r: (100.0 * regime_correct[r] / regime_total[r]) if regime_total[r] else None
                for r in regime_total
            },
            "regime_total": dict(regime_total),
        }
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default="outputs/result_trust_fusion.txt")
    ap.add_argument("--records", default="outputs/records_trust_fusion")
    ap.add_argument("--out", default="outputs/trust_fusion_report.md")
    args = ap.parse_args()

    if not Path(args.result).is_file():
        print(f"[ERROR] result file not found: {args.result}", file=sys.stderr)
        return 1

    parsed = parse_results(args.result)
    if not parsed:
        print(f"[ERROR] no parsable result lines in {args.result}", file=sys.stderr)
        return 1

    settings = sorted(parsed.keys(), key=lambda s: (s != CONTROL, s))
    datasets = sorted({d for ds in parsed.values() for d in ds.keys()})
    seeds = sorted({sd for ds in parsed.values() for d in ds.values() for sd in d.keys()})

    # ── Per-setting, per-dataset 4-seed mean ────────────────────────────────
    means = {}   # setting -> dataset -> mean acc
    for setting in settings:
        means[setting] = {}
        for dataset in datasets:
            accs = list(parsed[setting].get(dataset, {}).values())
            means[setting][dataset] = mean(accs) if accs else None

    control_means = means.get(CONTROL, {})

    lines = ["# Phase 1: Trust-Adaptive Read-Time Fusion — Results", ""]
    lines.append(f"Datasets: {', '.join(datasets)} | Seeds: {seeds}")
    lines.append("")
    lines.append("## Overall Accuracy (4-seed mean) vs. Control")
    lines.append("")
    header = "| Setting | " + " | ".join(datasets) + " | Avg | Δ Avg vs control |"
    lines.append(header)
    lines.append("|" + "---|" * (len(datasets) + 3))

    control_avg = mean([v for v in control_means.values() if v is not None]) if control_means else None

    verdicts = {}
    for setting in settings:
        row_vals = [means[setting].get(d) for d in datasets]
        row_avg = mean([v for v in row_vals if v is not None]) if any(v is not None for v in row_vals) else None
        delta_avg = (row_avg - control_avg) if (row_avg is not None and control_avg is not None) else None

        cells = [f"{v:.2f}" if v is not None else "N/A" for v in row_vals]
        avg_cell = f"{row_avg:.3f}" if row_avg is not None else "N/A"
        delta_cell = f"{delta_avg:+.3f}" if delta_avg is not None else "N/A"
        lines.append(f"| {setting} | " + " | ".join(cells) + f" | {avg_cell} | {delta_cell} |")

        # Go/no-go: >GAIN_BAR avg gain, no per-dataset regression beyond REGRESSION_BAND
        regressed = any(
            (means[setting].get(d) is not None and control_means.get(d) is not None
             and (means[setting][d] - control_means[d]) < -REGRESSION_BAND)
            for d in datasets
        )
        promote = (
            setting != CONTROL
            and delta_avg is not None
            and delta_avg > GAIN_BAR
            and not regressed
        )
        verdicts[setting] = promote

    lines.append("")
    lines.append(f"Go/no-go bar: avg gain > {GAIN_BAR}pp AND no single-dataset regression > {REGRESSION_BAND}pp vs. control.")
    lines.append("")
    lines.append("## Promotion Verdict")
    lines.append("")
    for setting in settings:
        if setting == CONTROL:
            continue
        verdict = "PROMOTE to held-out validation" if verdicts[setting] else "no-go"
        lines.append(f"- `{setting}`: **{verdict}**")
    lines.append("")

    # ── Tie-only + regime breakdown ─────────────────────────────────────────
    tq = tie_and_quadrant_stats(args.records, settings, datasets, seeds)
    lines.append("## Tie-Only Accuracy (pooled across datasets/seeds)")
    lines.append("")
    lines.append("| Setting | Tie N | Tie Acc |")
    lines.append("|---|---|---|")
    for setting in settings:
        t = tq[setting]
        acc = f"{t['tie_acc']:.1f}%" if t["tie_acc"] is not None else "N/A"
        lines.append(f"| {setting} | {t['tie_total']} | {acc} |")
    lines.append("")

    lines.append("## Accuracy by Trust Regime (pooled)")
    lines.append("")
    lines.append("| Setting | Trusted N | Trusted Acc | Untrusted N | Untrusted Acc |")
    lines.append("|---|---|---|---|---|")
    for setting in settings:
        t = tq[setting]
        rt = t["regime_total"].get("trusted", 0)
        ru = t["regime_total"].get("untrusted", 0)
        ra_t = t["regime_acc"].get("trusted")
        ra_u = t["regime_acc"].get("untrusted")
        lines.append(
            f"| {setting} | {rt} | {f'{ra_t:.1f}%' if ra_t is not None else 'N/A'} "
            f"| {ru} | {f'{ra_u:.1f}%' if ra_u is not None else 'N/A'} |"
        )
    lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] report written to {out_path}")

    n_promoted = sum(1 for v in verdicts.values() if v)
    print(f"[SUMMARY] {n_promoted}/{len(verdicts)} settings cleared the promotion bar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
