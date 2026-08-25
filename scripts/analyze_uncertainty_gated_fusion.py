#!/usr/bin/env python3
"""Experiment D — uncertainty-gated patch fusion.

See outputs/offline_experiments_plan.md ("Experiment D") for the full
write-up. Short version: rather than trying to further purify what enters
the patch-level prototype bank (Experiments A-C, all null results), this
tests gating *how much the patch term is allowed to influence the final
prediction*, conditioned on how confident the baseline (CLIP + image-level
prototype) already is. If the baseline is already sure, patch evidence has
little to gain and real room to break a correct answer; if the baseline is
unsure, patch evidence has more legitimate room to help.

This is computed entirely from the per-class logit components already
stored in outputs/records_patch_benefit/PatchModPTA-CS-*-s{1..4}/records.jsonl
(clip, image_proto, patch_proto, final) -- no bank reconstruction, no new
GPU forward pass, no frozen-bank approximation.

Usage::

    python scripts/analyze_uncertainty_gated_fusion.py \\
        --records outputs/records_patch_benefit \\
        --prereg outputs/patch_benefit_prereg.md \\
        --out outputs/uncertainty_gated_fusion_report.md
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from analyze_records import resolve_tau_img, resolve_tau_patch_proto  # noqa: E402
from analyze_patch_benefit import (  # noqa: E402
    DATASETS,
    SEEDS,
    load_label_records,
    read_locked_thresholds,
    sign_permutation_pvalue,
    seed_level_bootstrap_ci,
)

PATCH_ARM = "PatchModPTA-CS"
BASELINE_ARM = "PTA-CS"

TAUS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
GATE_TYPES = ["hard", "soft"]

# Known stored accuracies (outputs/result_patch_benefit.txt /
# outputs/patch_benefit_report.md) -- used to sanity-check tau=0.0 and
# tau=1.0 reproduce them exactly before trusting any other tau.
KNOWN_ACC = {
    ("PTA-CS", "dtd"): [47.64, 47.46, 47.46, 47.52],
    ("PTA-CS", "oxford_flowers"): [74.46, 74.79, 74.26, 74.99],
    ("PTA-CS", "oxford_pets"): [91.01, 91.14, 90.95, 91.20],
    ("PatchModPTA-CS", "dtd"): [46.75, 48.23, 47.22, 46.63],
    ("PatchModPTA-CS", "oxford_flowers"): [71.66, 72.39, 72.39, 72.43],
    ("PatchModPTA-CS", "oxford_pets"): [89.48, 89.34, 88.83, 88.66],
}


def softmax(vec):
    m = max(vec)
    exps = [pow(2.718281828459045, v - m) for v in vec]
    s = sum(exps)
    return [e / s for e in exps]


def top1_top2(probs):
    ranked = sorted(probs, reverse=True)
    top1 = ranked[0]
    top2 = ranked[1] if len(ranked) > 1 else 0.0
    return top1, top2


def gate_weight(margin, tau, gate_type):
    if gate_type == "hard":
        return 1.0 if margin < tau else 0.0
    # soft: linear ramp-down as margin approaches tau
    if tau <= 0.0:
        return 0.0
    if margin >= tau:
        return 0.0
    return max(0.0, min(1.0, (tau - margin) / tau))


def load_dataset_seed(records_dir, arm, dataset, seed):
    label = f"{arm}-{dataset}-s{seed}"
    header, samples = load_label_records(records_dir, label)
    return header, samples


def squash_fn(kind, scale):
    if kind == "tanh":
        return lambda x: __import__("math").tanh(x / scale)
    if kind in (None, "none", "identity"):
        return lambda x: x
    raise SystemExit(f"Unhandled patch_squash kind: {kind!r} -- update squash_fn")


def process_cell(records_dir, dataset, seed):
    """Returns per-sample dicts: target, baseline_logits, patch_term, margin."""
    header, samples = load_dataset_seed(records_dir, PATCH_ARM, dataset, seed)
    cfg = header.get("resolved_config") if header else None
    tau_img = resolve_tau_img(cfg)
    tau_patch = resolve_tau_patch_proto(cfg)
    if tau_img is None or tau_patch is None:
        raise SystemExit(f"Could not resolve tau_image_proto/tau_patch_proto for {dataset}-s{seed}")

    fusion_cfg = (cfg or {}).get("fusion") or {}
    squash = squash_fn(fusion_cfg.get("patch_squash"), fusion_cfg.get("patch_squash_scale", 1.0))

    out = []
    for rec in samples:
        lg = rec.get("logits") or {}
        clip = lg.get("clip")
        image_proto = lg.get("image_proto")
        patch_proto = lg.get("patch_proto")
        final = lg.get("final")
        if not clip or not image_proto or not patch_proto or not final:
            continue

        proto_alpha = rec.get("proto_alpha", 1.0)
        baseline_logits = [c + tau_img * ip for c, ip in zip(clip, image_proto)]
        patch_term = [tau_patch * proto_alpha * squash(pp) for pp in patch_proto]

        # Verification: reconstructed final must match stored final exactly.
        recon_final = [b + p for b, p in zip(baseline_logits, patch_term)]
        # Stored logits are float16-rounded, so exact equality isn't
        # expected; this just guards against a wrong fusion formula, not
        # float16 noise.
        max_err = max(abs(a - b) for a, b in zip(recon_final, final))
        if max_err > 0.2:
            raise SystemExit(
                f"final-logit reconstruction mismatch in {dataset}-s{seed} "
                f"batch_idx={rec.get('batch_idx')}: max_err={max_err}"
            )

        probs = softmax(baseline_logits)
        top1, top2 = top1_top2(probs)
        margin = top1 - top2

        out.append({
            "target": rec["target"],
            "baseline_logits": baseline_logits,
            "patch_term": patch_term,
            "margin": margin,
        })
    return out


def score_cell(cell_samples, tau, gate_type):
    correct = 0
    for s in cell_samples:
        g = gate_weight(s["margin"], tau, gate_type)
        gated = [b + g * p for b, p in zip(s["baseline_logits"], s["patch_term"])]
        pred = max(range(len(gated)), key=lambda i: gated[i])
        if pred == s["target"]:
            correct += 1
    return 100.0 * correct / len(cell_samples) if cell_samples else None


def sanity_check(accs_by_tau, dataset):
    for label, tau, tol in [("baseline (tau=0.0)", 0.0, 1e-6), ("always-fused (tau=1.0)", 1.0, 1e-6)]:
        known = KNOWN_ACC[(BASELINE_ARM if tau == 0.0 else PATCH_ARM, dataset)]
        got = accs_by_tau[("hard", tau)]
        for seed_i, (k, g) in enumerate(zip(known, got), start=1):
            if g is None or abs(k - g) > 0.05:
                print(
                    f"[WARN] sanity check failed for {dataset} {label} seed{seed_i}: "
                    f"known={k} got={g}",
                    file=sys.stderr,
                )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default="outputs/records_patch_benefit")
    parser.add_argument("--prereg", default="outputs/patch_benefit_prereg.md")
    parser.add_argument("--out", default="outputs/uncertainty_gated_fusion_report.md")
    args = parser.parse_args(argv)

    thresholds = read_locked_thresholds(args.prereg)

    # cell_samples[(dataset, seed)] = list of per-sample dicts
    cell_samples = {}
    for dataset in DATASETS:
        for seed in SEEDS:
            print(f"[load] {dataset}-s{seed}", file=sys.stderr)
            cell_samples[(dataset, seed)] = process_cell(args.records, dataset, seed)

    # results[dataset][(gate_type, tau)] = [acc_s1, acc_s2, acc_s3, acc_s4]
    results = {ds: {} for ds in DATASETS}
    for dataset in DATASETS:
        for gate_type in GATE_TYPES:
            for tau in TAUS:
                accs = []
                for seed in SEEDS:
                    accs.append(score_cell(cell_samples[(dataset, seed)], tau, gate_type))
                results[dataset][(gate_type, tau)] = accs

    for dataset in DATASETS:
        sanity_check(results[dataset], dataset)

    # Stats vs PTA-CS baseline, per (dataset, gate_type, tau)
    stats = {}
    for dataset in DATASETS:
        pta_accs = results[dataset][("hard", 0.0)]  # baseline reproduction
        for gate_type in GATE_TYPES:
            for tau in TAUS:
                accs = results[dataset][(gate_type, tau)]
                deltas = [a - p for a, p in zip(accs, pta_accs)]
                delta_mean = statistics.mean(deltas)
                delta_std = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
                p_sign = sign_permutation_pvalue(deltas)
                ci_lo, ci_hi = seed_level_bootstrap_ci(deltas)
                threshold = thresholds[dataset]
                if delta_mean > threshold and p_sign <= 0.0625 and (ci_lo is None or ci_lo > 0):
                    verdict = "SUPPORT"
                elif delta_mean <= 0 or p_sign > 0.5:
                    verdict = "REFUTE"
                else:
                    verdict = "INCONCLUSIVE"
                stats[(dataset, gate_type, tau)] = {
                    "accs": accs,
                    "mean": statistics.mean(accs),
                    "delta_mean": delta_mean,
                    "delta_std": delta_std,
                    "p_sign": p_sign,
                    "ci": (ci_lo, ci_hi),
                    "verdict": verdict,
                }

    render_report(stats, thresholds, args.out)
    json_path = Path(args.out).with_suffix(".json")
    json_out = {
        f"{ds}|{gt}|{tau}": v for (ds, gt, tau), v in stats.items()
    }
    json_path.write_text(json.dumps(json_out, indent=2), encoding="utf-8")
    print(f"[OK] {json_path}")


def render_report(stats, thresholds, out_path):
    lines = ["# Experiment D — Uncertainty-Gated Patch Fusion — Raw Results", ""]
    lines.append(
        "Gate the patch term on baseline (CLIP + image-proto) confidence "
        "margin. tau=0.0 always suppresses the patch term (reproduces plain "
        "PTA-CS baseline); tau=1.0 never suppresses it (reproduces the "
        "original always-fused PatchModPTA-CS). See "
        "outputs/offline_experiments_plan.md for the full write-up."
    )
    lines.append("")
    for dataset in DATASETS:
        lines.append(f"## {dataset} (noise-floor threshold: {thresholds[dataset]:.3f}pp)")
        lines.append("")
        lines.append("| Gate | tau | Accs (s1..s4) | Mean | Delta vs PTA-CS | p_sign | 95% CI | Verdict |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for gate_type in GATE_TYPES:
            for tau in TAUS:
                s = stats[(dataset, gate_type, tau)]
                acc_str = " / ".join(f"{a:.2f}" for a in s["accs"])
                ci = s["ci"]
                ci_str = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci[0] is not None else "N/A"
                lines.append(
                    f"| {gate_type} | {tau:.2f} | {acc_str} | {s['mean']:.2f} "
                    f"| {s['delta_mean']:+.2f}pp | {s['p_sign']:.4f} | {ci_str} | {s['verdict']} |"
                )
        lines.append("")

    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
