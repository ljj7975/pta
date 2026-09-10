#!/usr/bin/env python3
"""Wave-3 write-purity analysis.

Computes write-mask purity from per-sample records for the wave-3 methods
(AugPTA V8K4/V4K2, AnchorPTA d096/d093, ablations V1K1/V8K8) vs the archived
base-PTA baseline, testing each mechanism's core premise:

- AugPTA: V-view entropy-selected ensembling should raise pseudo-label
  quality, so write purity must rise if label quality is the active lever.
- AnchorPTA: writes damped by disagreement with the frozen text anchor
  should be less pure than undamped writes (the damping is selective),
  and damped writes should be more often wrong than the removed correct ones.

Write mask = softmax(logits.clip) >= 0.1 (PTA's write rule), recomputed
IDENTICALLY for every method from the stored per-sample logits: AugPTA's
logits.clip is the ensembled evidence (the mechanism's lever), while base PTA
and AnchorPTA store the single-view evidence. A write event is
(sample, class c) with mask[c]=1; write purity = P(c == target | event).
All methods are 4-seed means; the 2-seed sweep is excluded here.

For AnchorPTA the per-class cumulative n_damped counters let us split write
events into damped vs undamped (n_damped[c] increments exactly when class c's
write is damped), giving the purity of each group — the selectivity test.

Ablation V1K1 is the identity check: it must reproduce base-PTA purity
(and accuracy) exactly, confirming the GPU files are lossless reducible.

Usage: python scripts/wave3_write_purity.py
"""
import json
import os
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs_repr_upgrade")
ARCHIVE = os.path.join(REPO, "outputs_limitation_analysis_0faafb9")
DATASETS = ["dtd", "oxford_flowers", "oxford_pets"]
SEEDS = [1, 2, 3, 4]

RUN_DIRS = {
    "PTA-baseline": lambda ds, s: os.path.join(ARCHIVE, "records", f"PTA-{ds}-s{s}"),
    "AUG-V8K4": lambda ds, s: os.path.join(OUT, "records_aug_pta", f"AUG-V8K4-{ds}-s{s}"),
    "AUG-V4K2": lambda ds, s: os.path.join(OUT, "records_aug_pta", f"AUG-V4K2-{ds}-s{s}"),
    "ABL-V1K1": lambda ds, s: os.path.join(OUT, "records_abl", f"ABL-V1K1-{ds}-s{s}"),
    "ABL-V8K8": lambda ds, s: os.path.join(OUT, "records_abl", f"ABL-V8K8-{ds}-s{s}"),
    "ANCH-d096": lambda ds, s: os.path.join(OUT, "records_anchor_pta", f"ANCHOR-d096-{ds}-s{s}"),
    "ANCH-d093": lambda ds, s: os.path.join(OUT, "records_anchor_pta", f"ANCHOR-d093-{ds}-s{s}"),
}


def load_records(path):
    samples = []
    with open(os.path.join(path, "records.jsonl")) as fh:
        for line in fh:
            line = line.strip()
            if not line or '"__header__"' in line:
                continue
            samples.append(json.loads(line))
    return samples


def sample_metrics(rec, prev_damped):
    """Return (write_events, pure_events, damp_pure, damp_total) for a sample.

    write_events: classes c with softmax(clip)[c] >= 0.1 (PTA write mask).
    pure_events:  count of write events whose class == target.
    damp split (AnchorPTA only): a write event is damped iff class c's
    cumulative n_damped counter increased on this sample.
    """
    clip = np.array(rec["logits"]["clip"], float)
    probs = np.exp(clip - clip.max())
    probs /= probs.sum()
    mask = probs >= 1e-1
    target = int(rec["target"])
    events = mask.sum()
    pure = int(mask[target]) if target < len(mask) else 0

    stats = rec.get("proto_stats") or {}
    damp_pure = damp_total = 0
    # AnchorPTA keeps per-class counters: proto_stats["class_{c}"]["n_damped"].
    if "damp_threshold" in stats and events:
        for c in range(len(mask)):
            if not mask[c]:
                continue
            cls = stats.get("class_{}".format(c)) or {}
            cur = int(cls.get("n_damped", 0))
            if c in prev_damped and cur > prev_damped[c]:
                damp_total += 1
                damp_pure += int(c == target)
    return events, pure, damp_pure, damp_total


def run_stats(records):
    """Aggregate per-sample metrics into run-level stats."""
    n = len(records)
    tot_events = tot_pure = 0
    samples_with_write = 0
    damp_pure = damp_total = 0
    clip_correct = 0
    prev_damped = {}
    for rec in records:
        clip = np.array(rec["logits"]["clip"], float)
        clip_correct += int(clip.argmax() == int(rec["target"]))
        events, pure, dp, dt = sample_metrics(rec, prev_damped)
        tot_events += events
        tot_pure += pure
        damp_pure += dp
        damp_total += dt
        samples_with_write += int(events > 0)
        stats = rec.get("proto_stats") or {}
        if "damp_threshold" in stats:
            prev_damped = {
                int(k.rsplit("_", 1)[1]): int(v.get("n_damped", 0))
                for k, v in stats.items() if k.startswith("class_")
            }
    return {
        "n": n,
        "write_rate": 100.0 * samples_with_write / n,
        "purity": 100.0 * tot_pure / tot_events if tot_events else float("nan"),
        "damp_purity": 100.0 * damp_pure / damp_total if damp_total else float("nan"),
        "damp_rate": 100.0 * damp_total / tot_events if tot_events else float("nan"),
        "clip_acc": 100.0 * clip_correct / n,
    }


def main():
    print(f"{'method':12} {'ds':16} {'write%':>7} {'purity':>7} {'clip_acc':>9}")
    for method, make_dir in RUN_DIRS.items():
        per_ds = {}
        for ds in DATASETS:
            accs = []
            for s in SEEDS:
                path = make_dir(ds, s)
                if not os.path.isdir(path):
                    continue
                accs.append(run_stats(load_records(path)))
            if len(accs) < 4:
                continue
            wr = float(np.mean([a["write_rate"] for a in accs]))
            pu = float(np.mean([a["purity"] for a in accs]))
            ca = float(np.mean([a["clip_acc"] for a in accs]))
            per_ds[ds] = (wr, pu, ca)
            print(f"{method:12} {ds:16} {wr:7.2f} {pu:7.2f} {ca:9.2f}")
        if method.startswith("ANCH") and per_ds:
            print()
            print(f"  Anchor damp split (write events):")
            for ds in DATASETS:
                if ds not in per_ds:
                    continue
                dp = [run_stats(load_records(make_dir(ds, s)))["damp_purity"]
                      for s in SEEDS
                      if os.path.isdir(make_dir(ds, s))]
                dr = [run_stats(load_records(make_dir(ds, s)))["damp_rate"]
                      for s in SEEDS
                      if os.path.isdir(make_dir(ds, s))]
                if dp:
                    print(f"    {ds:16} damped events purity {np.mean(dp):7.2f}  "
                          f"damped/events {np.mean(dr):6.2f}%")


if __name__ == "__main__":
    main()