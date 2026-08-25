#!/usr/bin/env python3
"""Summarize Experiment B (per-cluster purity) + C (TCR accuracy) results
from scripts/frozen_bank_eval.py's per-dataset JSON outputs.

Usage::

    python scripts/summarize_frozen_bank_eval.py \\
        --eval-dir outputs/frozen_bank_eval \\
        --out outputs/frozen_bank_eval_report.md
"""

import argparse
import json
from pathlib import Path

DATASETS = ["dtd", "oxford_flowers", "oxford_pets"]


def appearance_tercile_purity(purity_table):
    n = len(purity_table)
    if n == 0:
        return None
    apps = sorted(r["appearance"] for r in purity_table)
    t1 = apps[n // 3]
    t2 = apps[2 * n // 3]
    buckets = {"low": [0.0, 0.0], "mid": [0.0, 0.0], "high": [0.0, 0.0]}
    for r in purity_table:
        b = "low" if r["appearance"] <= t1 else ("mid" if r["appearance"] <= t2 else "high")
        buckets[b][0] += r["touched"]
        buckets[b][1] += r["correct"]
    out = {}
    for b, (touched, correct) in buckets.items():
        out[b] = (100.0 * correct / touched) if touched else None
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", default="outputs/frozen_bank_eval")
    parser.add_argument("--out", default="outputs/frozen_bank_eval_report.md")
    args = parser.parse_args(argv)

    eval_dir = Path(args.eval_dir)
    summary = {}
    for ds in DATASETS:
        data = json.loads((eval_dir / f"{ds}.json").read_text())
        tbl = data["purity_table"]
        total_touched = sum(r["touched"] for r in tbl)
        total_correct = sum(r["correct"] for r in tbl)
        overall_purity = 100.0 * total_correct / total_touched if total_touched else None
        summary[ds] = {
            "n_total": data["n_total"],
            "acc_control": data["acc_control"],
            "acc_tcr": data["acc_tcr"],
            "delta_tcr": data["acc_tcr"] - data["acc_control"],
            "n_clusters_touched": len(tbl),
            "overall_purity": overall_purity,
            "tercile_purity": appearance_tercile_purity(tbl),
        }

    lines = ["# Frozen-Bank Evaluation Results (Experiments B + C)", ""]
    lines.append(
        "See `outputs/offline_experiments_plan.md` for the full method and "
        "caveats. Raw per-dataset output: `outputs/frozen_bank_eval/*.json`."
    )
    lines.append("")
    lines.append("## Experiment C: control (appearance weight) vs. TCR weight accuracy")
    lines.append("")
    lines.append("| Dataset | n | Control acc. | TCR acc. | Delta |")
    lines.append("|---|---|---|---|---|")
    for ds in DATASETS:
        s = summary[ds]
        lines.append(
            f"| {ds} | {s['n_total']} | {s['acc_control']:.2f}% | {s['acc_tcr']:.2f}% "
            f"| {s['delta_tcr']:+.2f}pp |"
        )
    lines.append("")

    lines.append("## Experiment B: does appearance weight track purity?")
    lines.append("")
    lines.append(
        "| Dataset | Clusters touched | Overall purity | Low-appearance tercile | "
        "Mid-appearance tercile | High-appearance tercile |"
    )
    lines.append("|---|---|---|---|---|---|")
    for ds in DATASETS:
        s = summary[ds]
        tp = s["tercile_purity"] or {}
        def f(v):
            return "N/A" if v is None else f"{v:.1f}%"
        lines.append(
            f"| {ds} | {s['n_clusters_touched']} | {f(s['overall_purity'])} "
            f"| {f(tp.get('low'))} | {f(tp.get('mid'))} | {f(tp.get('high'))} |"
        )
    lines.append("")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] {out_path}")

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] {json_path}")


if __name__ == "__main__":
    main()
