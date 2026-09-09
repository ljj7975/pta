#!/usr/bin/env python3
"""Phase 1a -- cheap offline diagnostic for class-prediction-frequency
calibration (see /home/brandon/.claude/plans/toasty-pondering-eagle.md).

Replays ALREADY-COMPLETED base-PTA records (outputs/records_trust_fusion/
TFP-control-<dataset>-s<seed>/records.jsonl -- control there is fusion tau
(1.0, 1.0), which reproduces plain PTA) and recomputes accuracy under a
class-prediction-frequency logit correction, using only the already-stored
`logits.final` array. No new GPU inference is performed.

Mechanism: maintain a causal running count of predicted classes seen so far
(from THIS replay's own re-derived predictions, not the original run's), and
subtract lam * log(running_freq / expected_freq + eps) from the final logits
before taking argmax. expected_freq = 1/C (datasets are close to balanced).

Usage:
    python scripts/check_prior_calibration_signal.py \
        --datasets dtd/oxford_flowers/oxford_pets --seeds 1,2,3,4 \
        --lams 0,0.05,0.1,0.2,0.5,1.0 \
        --out outputs/prior_calibration_diagnostic.md
"""
import argparse
import json
import math
import os


def load_records(path):
    header = None
    records = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("__header__"):
                header = obj
            else:
                records.append(obj)
    return header, records


def replay(records, C, lam, eps=1e-6):
    """Causal replay: running_freq updated from THIS replay's own predictions
    (not the original run's), so this is a fair simulation of what a live
    adapter applying the same correction would see."""
    counts = [0] * C
    n_seen = 0
    correct = 0
    expected = 1.0 / C
    for rec in records:
        final = rec["logits"]["final"]
        target = rec["target"]
        if lam > 0 and n_seen > 0:
            adjusted = [
                final[c] - lam * math.log(max(counts[c] / n_seen, eps) / expected + eps)
                for c in range(C)
            ]
        else:
            adjusted = final
        pred = max(range(C), key=lambda c: adjusted[c])
        counts[pred] += 1
        n_seen += 1
        if pred == target:
            correct += 1
    return 100.0 * correct / n_seen if n_seen else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="dtd/oxford_flowers/oxford_pets")
    ap.add_argument("--seeds", default="1,2,3,4")
    ap.add_argument("--lams", default="0,0.05,0.1,0.2,0.5,1.0")
    ap.add_argument("--record-root", default="outputs/records_trust_fusion")
    ap.add_argument("--label-prefix", default="TFP-control")
    ap.add_argument("--out", default="outputs/prior_calibration_diagnostic.md")
    args = ap.parse_args()

    datasets = args.datasets.split("/")
    seeds = [int(s) for s in args.seeds.split(",")]
    lams = [float(x) for x in args.lams.split(",")]

    lines = ["# Phase 1a: Class-Prediction-Frequency Calibration -- Offline Diagnostic", ""]
    lines.append(
        "Offline replay of already-completed base-PTA records "
        f"(`{args.record_root}/{args.label_prefix}-<dataset>-s<seed>/records.jsonl`). "
        "No new GPU inference. Causal running_freq derived from this replay's own predictions."
    )
    lines.append("")

    per_dataset_means = {}  # dataset -> {lam: mean_acc}
    for dataset in datasets:
        per_dataset_means[dataset] = {}
        for lam in lams:
            accs = []
            for seed in seeds:
                path = os.path.join(args.record_root, f"{args.label_prefix}-{dataset}-s{seed}", "records.jsonl")
                header, records = load_records(path)
                C = header["C"]
                acc = replay(records, C, lam)
                accs.append(acc)
            per_dataset_means[dataset][lam] = sum(accs) / len(accs)

    lines.append("## Accuracy (4-seed mean) vs. lam")
    lines.append("")
    header_row = "| lam | " + " | ".join(datasets) + " | Avg | Delta vs lam=0 |"
    lines.append(header_row)
    lines.append("|" + "---|" * (len(datasets) + 3))
    lam0_avg = sum(per_dataset_means[d][0.0] for d in datasets) / len(datasets)
    go_no_go = {}
    for lam in lams:
        vals = [per_dataset_means[d][lam] for d in datasets]
        avg = sum(vals) / len(vals)
        delta = avg - lam0_avg
        row = f"| {lam} | " + " | ".join(f"{v:.2f}" for v in vals) + f" | {avg:.3f} | {delta:+.3f} |"
        lines.append(row)
        if lam > 0:
            n_pass = sum(
                1 for d in datasets
                if (per_dataset_means[d][lam] - per_dataset_means[d][0.0]) > 0.3
            )
            go_no_go[lam] = n_pass

    lines.append("")
    lines.append("## Go/no-go check (per-dataset gain > 0.3pp vs lam=0)")
    lines.append("")
    lines.append("| lam | datasets clearing +0.3pp bar |")
    lines.append("|---|---|")
    any_promoted = False
    for lam, n_pass in go_no_go.items():
        lines.append(f"| {lam} | {n_pass}/{len(datasets)} |")
        if n_pass >= min(2, len(datasets)):
            any_promoted = True

    lines.append("")
    verdict = "PROCEED to Phase 1b" if any_promoted else "STOP -- do not build the calibration adapter"
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[OK] report written to {args.out}")


if __name__ == "__main__":
    main()
