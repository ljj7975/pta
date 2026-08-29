#!/usr/bin/env python3
"""Part 4 analysis — trust signal: does CLIP-patch vote agreement predict
classification quality?

Hypothesis: when CLIP and patch vote agree, CLIP's zero-shot prediction is
reliable. When they disagree, PTA's fused prototype-based prediction is
more trustworthy.  This script quantifies the trust signal by computing:

  1. Overall accuracy (PTA final prediction)
  2. Accuracy when CLIP and patch agree (trust CLIP)
  3. Accuracy when CLIP and patch disagree (trust PTA)
  4. Agreement rate and sample counts
  5. Comparison against PTA baseline

Reads JSONL records from:
  outputs/records/TrustSignal-{dataset}-s{seed}/records.jsonl

Outputs JSON to:
  outputs/trust_signal_analysis/{dataset}.json

Usage::

    python experiments/analysis/analyze_trust_signal.py --dataset dtd
    python experiments/analysis/analyze_trust_signal.py --dataset dtd --seeds 1,2,3,4
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

RECORDS_ROOT = REPO_ROOT / "outputs" / "records"
TRUST_RECORDS_PREFIX = "TrustSignal"
PTA_RECORDS_PREFIX = "PTA"
ANALYSIS_OUT = REPO_ROOT / "outputs" / "trust_signal_analysis"


def _to_float(val) -> float:
    """Coerce a JSON-loaded value to float (pyright-safe)."""
    return float(val)  # type: ignore[arg-type]


def argmax_idx(vec):
    """Index of the maximum value in a list, or None if empty."""
    if not vec:
        return None
    return max(range(len(vec)), key=lambda i: vec[i])


def load_records(record_dir):
    """Load per-sample JSONL records, skipping the header line.

    Returns:
        list of dicts (per-sample records).
    """
    path = record_dir / "records.jsonl"
    if not path.exists():
        return []
    samples = []
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


def compute_trust_signal_metrics(samples):
    """Compute trust-signal metrics from a list of per-sample records.

    For each sample:
      - clip_pred = argmax(record["logits"]["clip"])
      - patch_pred = argmax(record["logits"]["patch_proto"])
      - agrees = (clip_pred == patch_pred)
      - When agree:  trust CLIP → correct if clip_pred == target
      - When disagree: trust PTA → correct if record["pred"] == target

    Returns a dict of metrics.
    """
    n_total = 0
    n_agree = 0
    n_disagree = 0
    agree_correct = 0
    disagree_correct = 0
    overall_correct = 0

    for rec in samples:
        logits = rec.get("logits") or {}
        clip_logits = logits.get("clip")
        patch_proto_logits = logits.get("patch_proto")

        if not clip_logits or not patch_proto_logits:
            continue

        clip_pred = argmax_idx(clip_logits)
        patch_pred = argmax_idx(patch_proto_logits)
        target = rec.get("target")
        pta_pred = rec.get("pred")

        if clip_pred is None or patch_pred is None or target is None:
            continue

        n_total += 1
        agrees = (clip_pred == patch_pred)

        if agrees:
            n_agree += 1
            if clip_pred == target:
                agree_correct += 1
        else:
            n_disagree += 1
            if pta_pred is not None and pta_pred == target:
                disagree_correct += 1

        if rec.get("correct"):
            overall_correct += 1

    acc_overall = 100.0 * overall_correct / n_total if n_total else 0.0
    acc_agree = 100.0 * agree_correct / n_agree if n_agree else 0.0
    acc_disagree = 100.0 * disagree_correct / n_disagree if n_disagree else 0.0
    agreement_rate = 100.0 * n_agree / n_total if n_total else 0.0

    # Trust-signal combined: use CLIP when agree, PTA when disagree
    trust_correct = agree_correct + disagree_correct
    acc_trust = 100.0 * trust_correct / n_total if n_total else 0.0

    return {
        "n_total": n_total,
        "n_agree": n_agree,
        "n_disagree": n_disagree,
        "overall_acc": round(acc_overall, 4),
        "agree_acc": round(acc_agree, 4),
        "disagree_acc": round(acc_disagree, 4),
        "agreement_rate": round(agreement_rate, 4),
        "trust_signal_acc": round(acc_trust, 4),
        "agree_correct": agree_correct,
        "disagree_correct": disagree_correct,
        "overall_correct": overall_correct,
    }


def compute_per_seed_metrics(dataset, seeds):
    """Compute trust-signal metrics per seed and aggregate across seeds."""
    per_seed = {}
    for seed in seeds:
        record_dir = RECORDS_ROOT / f"{TRUST_RECORDS_PREFIX}-{dataset}-s{seed}"
        samples = load_records(record_dir)
        if not samples:
            print(f"  [skip] {record_dir} not found or empty", file=sys.stderr)
            continue
        metrics = compute_trust_signal_metrics(samples)
        per_seed[seed] = metrics

    if not per_seed:
        return None, per_seed

    # Aggregate across seeds
    seeds_used = sorted(per_seed.keys())
    agg = {
        "n_total": 0,
        "n_agree": 0,
        "n_disagree": 0,
        "agree_correct": 0,
        "disagree_correct": 0,
        "overall_correct": 0,
    }
    for s in seeds_used:
        m = per_seed[s]
        agg["n_total"] += m["n_total"]
        agg["n_agree"] += m["n_agree"]
        agg["n_disagree"] += m["n_disagree"]
        agg["agree_correct"] += m["agree_correct"]
        agg["disagree_correct"] += m["disagree_correct"]
        agg["overall_correct"] += m["overall_correct"]

    n = agg["n_total"]
    na = agg["n_agree"]
    nd = agg["n_disagree"]
    aggregated = {
        "seeds_used": seeds_used,
        "n_total": n,
        "n_agree": na,
        "n_disagree": nd,
        "overall_acc": round(100.0 * agg["overall_correct"] / n, 4) if n else 0.0,
        "agree_acc": round(100.0 * agg["agree_correct"] / na, 4) if na else 0.0,
        "disagree_acc": round(100.0 * agg["disagree_correct"] / nd, 4) if nd else 0.0,
        "agreement_rate": round(100.0 * na / n, 4) if n else 0.0,
        "trust_signal_acc": round(
            100.0 * (agg["agree_correct"] + agg["disagree_correct"]) / n, 4
        ) if n else 0.0,
    }

    return aggregated, per_seed


def compute_pta_baseline(dataset, seeds):
    """Load PTA baseline records and compute overall accuracy for comparison."""
    pta_accs = []
    pta_ns = []
    for seed in seeds:
        record_dir = RECORDS_ROOT / f"{PTA_RECORDS_PREFIX}-{dataset}-s{seed}"
        samples = load_records(record_dir)
        if not samples:
            continue
        n = len(samples)
        correct = sum(1 for s in samples if s.get("correct"))
        pta_accs.append(100.0 * correct / n if n else 0.0)
        pta_ns.append(n)

    if not pta_accs:
        return None

    total_n = sum(pta_ns)
    total_correct = sum(int(a * n / 100.0) for a, n in zip(pta_accs, pta_ns))
    return {
        "overall_acc": round(sum(pta_accs) / len(pta_accs), 4),
        "seeds_available": len(pta_accs),
        "n_total": total_n,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True,
                        choices=["dtd", "oxford_flowers", "oxford_pets"],
                        help="Dataset to analyze")
    parser.add_argument("--seeds", default="1,2,3,4",
                        help="Comma-separated seeds (default: 1,2,3,4)")
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",")]
    dataset = args.dataset

    print(f"Analyzing trust signal for {dataset}, seeds={seeds}")

    aggregated, per_seed = compute_per_seed_metrics(dataset, seeds)

    if aggregated is None:
        print(f"[ERROR] No TrustSignal records found for {dataset}",
              file=sys.stderr)
        return 1

    # PTA baseline comparison
    pta_baseline = compute_pta_baseline(dataset, seeds)

    # Build output
    result = {
        "dataset": dataset,
        "method": "TrustSignal",
        "description": (
            "Trust-signal analysis: when CLIP and patch vote agree, "
            "trust CLIP; when they disagree, trust PTA's fused prediction."
        ),
        "aggregated": aggregated,
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "pta_baseline": pta_baseline,
    }

    # Add delta if both trust signal and PTA baseline are available
    if pta_baseline is not None and aggregated is not None:
        trust_acc = _to_float(aggregated["trust_signal_acc"])
        pta_acc = _to_float(pta_baseline["overall_acc"])
        result["delta_vs_pta"] = round(trust_acc - pta_acc, 4)

    # Write output
    out_dir = ANALYSIS_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Print summary
    print(f"\n--- {dataset} trust signal summary ---")
    print(f"  Overall accuracy (PTA final): {aggregated['overall_acc']:.2f}%")
    print(f"  Trust signal accuracy:        {aggregated['trust_signal_acc']:.2f}%")
    print(f"  Agreement rate:               {aggregated['agreement_rate']:.2f}%")
    print(f"  When agree  (n={aggregated['n_agree']}): acc = {aggregated['agree_acc']:.2f}%")
    print(f"  When disagree (n={aggregated['n_disagree']}): acc = {aggregated['disagree_acc']:.2f}%")
    if pta_baseline:
        print(f"  PTA baseline:                 {pta_baseline['overall_acc']:.2f}%")
        print(f"  Delta vs PTA:                 {result.get('delta_vs_pta', 0):+.2f} pp")

    print(f"\n[OK] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
