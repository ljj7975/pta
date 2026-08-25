#!/usr/bin/env python3
"""Patch-level prototype benefit study — accuracy, tie-breaking, and
flip-metrics analysis across all 7 arms x 3 datasets x 4 seeds.

See outputs/patch_benefit_prereg.md for the pre-registered hypothesis and
decision rule this script evaluates. This script does not choose or tune
that rule — it only computes the numbers the rule is applied to.

Usage::

    # Stage B lock: compute PTA-CS noise floor and append to the prereg file.
    # Run this once PTA-CS seeds 1-4 exist for all 3 datasets, BEFORE
    # inspecting any patch-arm number.
    python scripts/analyze_patch_benefit.py --records outputs/records_patch_benefit \\
        --prereg outputs/patch_benefit_prereg.md --lock-noise-floor

    # Sanity-check record counts / header seed consistency only.
    python scripts/analyze_patch_benefit.py --records outputs/records_patch_benefit \\
        --verify-only

    # Full report (requires the Stage B lock to already be present).
    python scripts/analyze_patch_benefit.py --records outputs/records_patch_benefit \\
        --prereg outputs/patch_benefit_prereg.md \\
        --out outputs/patch_benefit_report.md \\
        --verdict-json outputs/patch_benefit_verdict.json
"""

import argparse
import itertools
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from analyze_records import (  # noqa: E402
    argmax_idx,
    flip_metrics_v2,
    resolve_tau_img,
    resolve_tau_patch_proto,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

# ---------------------------------------------------------------------------
# Study constants
# ---------------------------------------------------------------------------

DATASETS = ["dtd", "oxford_flowers", "oxford_pets"]
SEEDS = [1, 2, 3, 4]
BASELINE_ARM = "PTA-CS"

# Longest-match-first: PatchModPTA-CS-QGated must be checked before
# PatchModPTA-CS, or the shorter prefix would swallow it.
ARM_PREFIXES = [
    "PatchModPTA-CS-QGated",
    "PatchModPTA-CS-MVote",
    "PatchModPTA-CS-AGate",
    "PatchModPTA-CS",
    "PTA-CS",
    "ratio-pta-mv",
    "ratio-clip-mv",
]
ARM_PREFIXES_SORTED = sorted(ARM_PREFIXES, key=len, reverse=True)

PATCH_ARMS = [a for a in ARM_PREFIXES if a != BASELINE_ARM]

EXPECTED_SAMPLES = {"dtd": 1692, "oxford_flowers": 2463, "oxford_pets": 3669}

DECISION_DELTA_FLOOR_PP = 1.0
NOISE_MULTIPLIER = 3.0
SIGN_P_SUPPORT = 0.0625
SIGN_P_REFUTE = 0.5
N_RESAMPLES = 10000
BOOTSTRAP_SEED = 0

_LABEL_RE = re.compile(r"^(?P<dataset>[A-Za-z_0-9]+)-s(?P<seed>\d+)$")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def list_labels(records_dir):
    d = Path(records_dir)
    if not d.is_dir():
        return []
    return sorted(
        sub.name for sub in d.iterdir()
        if sub.is_dir() and (sub / "records.jsonl").is_file()
    )


def load_label_records(records_dir, label):
    path = Path(records_dir) / label / "records.jsonl"
    header = None
    samples = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("__header__"):
                header = rec
            else:
                samples.append(rec)
    return header, samples


def parse_label(label):
    """label -> (arm, dataset, seed) or (None, None, None) if unrecognized."""
    for prefix in ARM_PREFIXES_SORTED:
        if label.startswith(prefix + "-"):
            rest = label[len(prefix) + 1:]
            m = _LABEL_RE.match(rest)
            if m:
                return prefix, m.group("dataset"), int(m.group("seed"))
    return None, None, None


def load_all(records_dir):
    """{(arm, dataset, seed): (header, samples)}"""
    idx = {}
    for label in list_labels(records_dir):
        arm, dataset, seed = parse_label(label)
        if arm is None or dataset not in DATASETS or seed not in SEEDS:
            continue
        header, samples = load_label_records(records_dir, label)
        idx[(arm, dataset, seed)] = (header, samples)
    return idx


def samples_by_batch(samples):
    return {s["batch_idx"]: s for s in samples}


def cell_accuracy(samples):
    if not samples:
        return None
    return 100.0 * sum(1 for s in samples if s.get("correct")) / len(samples)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(index, arms):
    problems = []
    for arm in arms:
        for dataset in DATASETS:
            for seed in SEEDS:
                key = (arm, dataset, seed)
                if key not in index:
                    problems.append(f"MISSING: {arm}-{dataset}-s{seed}")
                    continue
                header, samples = index[key]
                expected = EXPECTED_SAMPLES[dataset]
                if len(samples) != expected:
                    problems.append(
                        f"SAMPLE COUNT MISMATCH: {arm}-{dataset}-s{seed} "
                        f"has {len(samples)} samples, expected {expected}"
                    )
                header_seed = header.get("seed") if header else None
                if header_seed != seed:
                    problems.append(
                        f"WARN header-seed mismatch (known SEED-env bug): "
                        f"{arm}-{dataset}-s{seed} header says seed={header_seed} "
                        f"-- trusting label-derived seed={seed}"
                    )
    return problems


# ---------------------------------------------------------------------------
# Stage B noise-floor lock
# ---------------------------------------------------------------------------

def compute_noise_floor(index):
    """{dataset: (sigma_pta, threshold, accs)}"""
    out = {}
    for dataset in DATASETS:
        accs = []
        for seed in SEEDS:
            key = (BASELINE_ARM, dataset, seed)
            if key not in index:
                raise SystemExit(
                    f"Cannot lock noise floor: missing {BASELINE_ARM}-{dataset}-s{seed}"
                )
            _, samples = index[key]
            accs.append(cell_accuracy(samples))
        sigma = statistics.stdev(accs)  # ddof=1, n=4
        threshold = max(DECISION_DELTA_FLOOR_PP, NOISE_MULTIPLIER * sigma)
        out[dataset] = (sigma, threshold, accs)
    return out


def lock_noise_floor(records_dir, prereg_path):
    index = load_all(records_dir)
    problems = verify(index, [BASELINE_ARM])
    hard_fails = [p for p in problems if not p.startswith("WARN")]
    if hard_fails:
        print("Cannot lock noise floor -- verification failed:", file=sys.stderr)
        for p in hard_fails:
            print(f"  {p}", file=sys.stderr)
        raise SystemExit(1)
    for p in problems:
        if p.startswith("WARN"):
            print(p, file=sys.stderr)

    floor = compute_noise_floor(index)

    from datetime import datetime, timezone
    lines = [
        "",
        f"### Stage B lock — {datetime.now(timezone.utc).isoformat()}",
        "",
        "Computed from PTA-CS seeds 1-4, before any patch-arm accuracy was inspected.",
        "",
        "| Dataset | PTA-CS accuracies (s1..s4) | sigma_PTA (ddof=1) | threshold(dataset) |",
        "|---|---|---|---|",
    ]
    for dataset in DATASETS:
        sigma, threshold, accs = floor[dataset]
        acc_str = " / ".join(f"{a:.2f}" for a in accs)
        lines.append(f"| {dataset} | {acc_str} | {sigma:.3f}pp | {threshold:.3f}pp |")
    lines.append("")

    text = Path(prereg_path).read_text(encoding="utf-8")
    marker = "<!-- STAGE_B_LOCK: pending -->"
    if marker not in text:
        raise SystemExit(
            f"Stage B marker not found in {prereg_path} -- "
            "already locked, or prereg file was edited unexpectedly."
        )
    text = text.replace(marker, "<!-- STAGE_B_LOCK: done -->\n" + "\n".join(lines))
    Path(prereg_path).write_text(text, encoding="utf-8")
    print(f"[OK] Stage B noise floor locked into {prereg_path}")
    for dataset in DATASETS:
        sigma, threshold, accs = floor[dataset]
        print(f"  {dataset}: sigma={sigma:.3f}pp threshold={threshold:.3f}pp")
    return floor


def read_locked_thresholds(prereg_path):
    text = Path(prereg_path).read_text(encoding="utf-8")
    if "<!-- STAGE_B_LOCK: done -->" not in text:
        raise SystemExit(
            f"{prereg_path} has no Stage B lock -- run --lock-noise-floor first. "
            "Refusing to compute a verdict without a pre-registered noise floor."
        )
    out = {}
    for m in re.finditer(
        r"\|\s*(\w+)\s*\|[^|]*\|\s*([\d.]+)pp\s*\|\s*([\d.]+)pp\s*\|", text
    ):
        dataset, sigma, threshold = m.group(1), float(m.group(2)), float(m.group(3))
        if dataset in DATASETS:
            out[dataset] = threshold
    missing = set(DATASETS) - set(out)
    if missing:
        raise SystemExit(f"Stage B lock missing thresholds for: {missing}")
    return out


# ---------------------------------------------------------------------------
# Accuracy table + paired statistics
# ---------------------------------------------------------------------------

def compute_accuracy_table(index, arms):
    table = {}
    for arm in arms:
        for dataset in DATASETS:
            cell_accs = []
            for seed in SEEDS:
                key = (arm, dataset, seed)
                if key in index:
                    _, samples = index[key]
                    cell_accs.append(cell_accuracy(samples))
                else:
                    cell_accs.append(None)
            valid = [a for a in cell_accs if a is not None]
            table[(arm, dataset)] = {
                "accs": cell_accs,
                "mean": statistics.mean(valid) if valid else None,
                "std": statistics.stdev(valid) if len(valid) > 1 else None,
            }
    return table


def sign_permutation_pvalue(deltas):
    n = len(deltas)
    obs_mean = sum(deltas) / n
    abs_deltas = [abs(d) for d in deltas]
    total = 0
    count_ge = 0
    for signs in itertools.product([1, -1], repeat=n):
        perm_mean = sum(s * a for s, a in zip(signs, abs_deltas)) / n
        total += 1
        if perm_mean >= obs_mean - 1e-9:
            count_ge += 1
    return count_ge / total


def seed_level_bootstrap_ci(deltas, n_resamples=N_RESAMPLES, seed=BOOTSTRAP_SEED):
    if np is None:
        return None, None
    rng = np.random.default_rng(seed)
    arr = np.asarray(deltas, dtype=float)
    n = len(arr)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    means.sort()
    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples) - 1]
    return float(lo), float(hi)


def sample_level_paired_bootstrap_ci(pta_samples, arm_samples, n_resamples=N_RESAMPLES, seed=BOOTSTRAP_SEED):
    if np is None:
        return None, None
    pta_by_idx = samples_by_batch(pta_samples)
    arm_by_idx = samples_by_batch(arm_samples)
    common = sorted(set(pta_by_idx) & set(arm_by_idx))
    if not common:
        return None, None
    pairs = np.array(
        [
            (1 if arm_by_idx[i].get("correct") else 0)
            - (1 if pta_by_idx[i].get("correct") else 0)
            for i in common
        ],
        dtype=float,
    )
    n = len(pairs)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = 100.0 * pairs[idx].mean(axis=1)
    means.sort()
    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples) - 1]
    return float(lo), float(hi)


def compute_deltas_and_stats(index, arms):
    """{(arm, dataset): {deltas, delta_mean, delta_std, p_sign, seed_ci, sample_ci_by_seed}}"""
    out = {}
    for arm in arms:
        for dataset in DATASETS:
            deltas = []
            sample_cis = []
            for seed in SEEDS:
                pta_key = (BASELINE_ARM, dataset, seed)
                arm_key = (arm, dataset, seed)
                if pta_key not in index or arm_key not in index:
                    deltas.append(None)
                    sample_cis.append((None, None))
                    continue
                _, pta_samples = index[pta_key]
                _, arm_samples = index[arm_key]
                d = cell_accuracy(arm_samples) - cell_accuracy(pta_samples)
                deltas.append(d)
                sample_cis.append(
                    sample_level_paired_bootstrap_ci(pta_samples, arm_samples)
                )
            valid = [d for d in deltas if d is not None]
            if len(valid) < len(SEEDS):
                out[(arm, dataset)] = {
                    "deltas": deltas, "delta_mean": None, "delta_std": None,
                    "p_sign": None, "seed_ci": (None, None), "sample_cis": sample_cis,
                }
                continue
            delta_mean = statistics.mean(valid)
            delta_std = statistics.stdev(valid)
            p_sign = sign_permutation_pvalue(valid)
            seed_ci = seed_level_bootstrap_ci(valid)
            out[(arm, dataset)] = {
                "deltas": deltas, "delta_mean": delta_mean, "delta_std": delta_std,
                "p_sign": p_sign, "seed_ci": seed_ci, "sample_cis": sample_cis,
            }
    return out


# ---------------------------------------------------------------------------
# Tie / agreement breakdown (generalizes outputs/tie_breaking_analysis.md
# from dtd-seed-1-only to all 3 datasets x 4 seeds x 6 patch arms)
# ---------------------------------------------------------------------------

def tie_breakdown_cell(pta_samples, arm_samples):
    """Pooled-over-nothing (single seed) tie stats for one (arm, dataset, seed)."""
    pta_by_idx = samples_by_batch(pta_samples)
    arm_by_idx = samples_by_batch(arm_samples)
    common = sorted(set(pta_by_idx) & set(arm_by_idx))

    n_total = len(common)
    tie_idxs = []
    for i in common:
        lg = pta_by_idx[i].get("logits") or {}
        clip, img = lg.get("clip"), lg.get("image_proto")
        if not clip or not img:
            continue
        if argmax_idx(clip) != argmax_idx(img):
            tie_idxs.append(i)

    n_ties = len(tie_idxs)
    pta_correct_tie = sum(1 for i in tie_idxs if pta_by_idx[i].get("correct"))
    arm_correct_tie = sum(1 for i in tie_idxs if arm_by_idx[i].get("correct"))

    patch_helps = patch_hurts = both_right = both_wrong = 0
    buckets = {"image_proto": 0, "clip": 0, "other": 0}
    bucket_correct = {"image_proto": 0, "clip": 0, "other": 0}
    gt_match = 0
    n_patch_available = 0

    for i in tie_idxs:
        p, a = pta_by_idx[i], arm_by_idx[i]
        pta_ok, arm_ok = bool(p.get("correct")), bool(a.get("correct"))
        if arm_ok and not pta_ok:
            patch_helps += 1
        elif not arm_ok and pta_ok:
            patch_hurts += 1
        elif arm_ok and pta_ok:
            both_right += 1
        else:
            both_wrong += 1

        lg = p.get("logits") or {}
        clip, img = lg.get("clip"), lg.get("image_proto")
        c_idx, im_idx = argmax_idx(clip), argmax_idx(img)
        a_lg = a.get("logits") or {}
        patch = a_lg.get("patch_proto")
        if not patch:
            continue
        p_idx = argmax_idx(patch)
        if p_idx is None:
            continue
        n_patch_available += 1
        target = p.get("target")

        if p_idx == im_idx:
            bucket = "image_proto"
            source_correct = im_idx == target
        elif p_idx == c_idx:
            bucket = "clip"
            source_correct = c_idx == target
        else:
            bucket = "other"
            source_correct = p_idx == target
        buckets[bucket] += 1
        if source_correct:
            bucket_correct[bucket] += 1
        if p_idx == target:
            gt_match += 1

    return {
        "n_total": n_total,
        "n_ties": n_ties,
        "pta_correct_tie": pta_correct_tie,
        "arm_correct_tie": arm_correct_tie,
        "patch_helps": patch_helps,
        "patch_hurts": patch_hurts,
        "both_right": both_right,
        "both_wrong": both_wrong,
        "buckets": buckets,
        "bucket_correct": bucket_correct,
        "gt_match": gt_match,
        "n_patch_available": n_patch_available,
    }


def _sum_cells(cells):
    out = {
        "n_total": 0, "n_ties": 0, "pta_correct_tie": 0, "arm_correct_tie": 0,
        "patch_helps": 0, "patch_hurts": 0, "both_right": 0, "both_wrong": 0,
        "buckets": {"image_proto": 0, "clip": 0, "other": 0},
        "bucket_correct": {"image_proto": 0, "clip": 0, "other": 0},
        "gt_match": 0, "n_patch_available": 0,
    }
    for c in cells:
        for k in ("n_total", "n_ties", "pta_correct_tie", "arm_correct_tie",
                   "patch_helps", "patch_hurts", "both_right", "both_wrong",
                   "gt_match", "n_patch_available"):
            out[k] += c[k]
        for bucket in out["buckets"]:
            out["buckets"][bucket] += c["buckets"][bucket]
            out["bucket_correct"][bucket] += c["bucket_correct"][bucket]
    return out


def compute_tie_breakdown(index, arms):
    """{(arm, dataset): pooled-over-4-seeds tie stats}"""
    out = {}
    for arm in arms:
        for dataset in DATASETS:
            cells = []
            for seed in SEEDS:
                pta_key, arm_key = (BASELINE_ARM, dataset, seed), (arm, dataset, seed)
                if pta_key not in index or arm_key not in index:
                    continue
                _, pta_samples = index[pta_key]
                _, arm_samples = index[arm_key]
                cells.append(tie_breakdown_cell(pta_samples, arm_samples))
            if cells:
                out[(arm, dataset)] = _sum_cells(cells)
    return out


# ---------------------------------------------------------------------------
# flip_metrics_v2 aggregation
# ---------------------------------------------------------------------------

def _sum_flip_component(comps):
    keys = set()
    for c in comps:
        keys.update(c.keys())
    out = {}
    for k in ("n", "correct", "corr", "reg", "masked"):
        if any(k in c for c in comps):
            out[k] = sum(c.get(k, 0) for c in comps)
    return out


def compute_flip_aggregate(index, arms):
    """{(arm, dataset): pooled flip_metrics_v2 aggregate over 4 seeds}"""
    out = {}
    for arm in arms:
        for dataset in DATASETS:
            text_only, image, patch_alone, patch_image = [], [], [], []
            for seed in SEEDS:
                key = (arm, dataset, seed)
                if key not in index:
                    continue
                header, samples = index[key]
                cfg = (header or {}).get("resolved_config") or {}
                tau_img = resolve_tau_img(cfg)
                tau_patch = resolve_tau_patch_proto(cfg)
                fm = flip_metrics_v2(samples, tau_img=tau_img, tau_patch_proto=tau_patch)
                agg = fm["aggregate"]
                text_only.append(agg["text_only"])
                image.append(agg["image"])
                patch_alone.append(agg["patch_alone"])
                patch_image.append(agg["patch_image"])
            if not text_only:
                continue

            def finalize_acc(comps):
                s = _sum_flip_component(comps)
                n, c = s.get("n", 0), s.get("correct", 0)
                s["acc"] = 100.0 * c / n if n else None
                return s

            def finalize_flip(comps):
                s = _sum_flip_component(comps)
                corr, reg = s.get("corr", 0), s.get("reg", 0)
                denom = corr + reg
                s["acc"] = 100.0 * corr / denom if denom else None
                s["net"] = corr - reg
                return s

            out[(arm, dataset)] = {
                "text_only": finalize_acc(text_only),
                "image": finalize_flip(image),
                "patch_alone": finalize_flip(patch_alone),
                "patch_image": finalize_flip(patch_image),
            }
    return out


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def apply_verdict(deltas_stats, flip_agg, thresholds, arms):
    cell_verdicts = {}
    for arm in arms:
        for dataset in DATASETS:
            ds = deltas_stats.get((arm, dataset))
            if not ds or ds["delta_mean"] is None:
                cell_verdicts[(arm, dataset)] = "MISSING_DATA"
                continue
            delta_mean, p_sign = ds["delta_mean"], ds["p_sign"]
            ci_lo, _ = ds["seed_ci"]
            threshold = thresholds[dataset]
            flip = flip_agg.get((arm, dataset), {})
            patch_image_net = (flip.get("patch_image") or {}).get("net")

            if delta_mean <= 0 or p_sign > SIGN_P_REFUTE:
                cell_verdicts[(arm, dataset)] = "REFUTE"
            elif (
                delta_mean > threshold
                and p_sign <= SIGN_P_SUPPORT
                and ci_lo is not None and ci_lo > 0
                and patch_image_net is not None and patch_image_net > 0
            ):
                cell_verdicts[(arm, dataset)] = "SUPPORT"
            else:
                cell_verdicts[(arm, dataset)] = "INCONCLUSIVE"

    support_datasets = defaultdict(set)
    for (arm, dataset), v in cell_verdicts.items():
        if v == "SUPPORT":
            support_datasets[arm].add(dataset)

    qualifying_arms = [a for a in arms if len(support_datasets.get(a, set())) >= 2]
    non_ratio_pta_support = any(
        a != "ratio-pta-mv" and support_datasets.get(a) for a in arms
    )

    if qualifying_arms and (
        any(a != "ratio-pta-mv" for a in qualifying_arms) or non_ratio_pta_support
    ):
        overall = "SUPPORT"
    elif not any(support_datasets.values()):
        overall = "REFUTE"
    else:
        overall = "INCONCLUSIVE"

    return cell_verdicts, overall


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_report(acc_table, deltas_stats, tie_breakdown, flip_agg, cell_verdicts,
                   overall_verdict, thresholds, arms):
    lines = ["# Patch-Level Prototype Benefit Study — Report", ""]
    lines.append("Scope: dtd, oxford_flowers, oxford_pets — ViT-B/16, CLIP Surgery, seeds 1-4.")
    lines.append("Decision rule: see `outputs/patch_benefit_prereg.md`.")
    lines.append("")

    lines.append("## 1. Accuracy Table")
    lines.append("")
    lines.append("| Arm | " + " | ".join(DATASETS) + " |")
    lines.append("|---|" + "---|" * len(DATASETS))
    for arm in [BASELINE_ARM] + arms:
        cells = []
        for dataset in DATASETS:
            t = acc_table.get((arm, dataset), {})
            if t.get("mean") is None:
                cells.append("N/A")
                continue
            accs_str = "/".join(f"{a:.2f}" if a is not None else "NA" for a in t["accs"])
            std_str = f"{t['std']:.2f}" if t["std"] is not None else "NA"
            cells.append(f"{t['mean']:.2f}±{std_str} [{accs_str}]")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 2. Paired Delta vs PTA-CS")
    lines.append("")
    lines.append("| Arm | Dataset | deltas (s1..s4) | mean | p_sign | seed CI | threshold | verdict |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for arm in arms:
        for dataset in DATASETS:
            ds = deltas_stats.get((arm, dataset), {})
            if ds.get("delta_mean") is None:
                lines.append(f"| {arm} | {dataset} | N/A | N/A | N/A | N/A | N/A | MISSING_DATA |")
                continue
            deltas_str = "/".join(f"{d:+.2f}" if d is not None else "NA" for d in ds["deltas"])
            lo, hi = ds["seed_ci"]
            ci_str = f"[{lo:+.2f}, {hi:+.2f}]" if lo is not None else "N/A"
            v = cell_verdicts.get((arm, dataset), "?")
            lines.append(
                f"| {arm} | {dataset} | {deltas_str} | {ds['delta_mean']:+.2f}pp | "
                f"{ds['p_sign']:.4f} | {ci_str} | {thresholds[dataset]:.2f}pp | **{v}** |"
            )
    lines.append("")

    lines.append("## 3. Tie / Agreement Breakdown (generalizes outputs/tie_breaking_analysis.md)")
    lines.append("")
    lines.append("A tie is `clip.argmax != image_proto.argmax` (per PTA-CS's own logits, pooled over 4 seeds).")
    lines.append("")
    lines.append("| Arm | Dataset | ties/total | PTA acc on ties | arm acc on ties | helps | hurts | agree-image | agree-clip | other | patch-GT-match |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in arms:
        for dataset in DATASETS:
            tb = tie_breakdown.get((arm, dataset))
            if not tb or tb["n_ties"] == 0:
                lines.append(f"| {arm} | {dataset} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
                continue
            n_ties = tb["n_ties"]
            tie_rate = 100.0 * n_ties / tb["n_total"] if tb["n_total"] else 0.0
            pta_acc = 100.0 * tb["pta_correct_tie"] / n_ties
            arm_acc = 100.0 * tb["arm_correct_tie"] / n_ties
            b = tb["buckets"]
            npatch = tb["n_patch_available"] or 1
            gt_rate = 100.0 * tb["gt_match"] / npatch
            lines.append(
                f"| {arm} | {dataset} | {n_ties}/{tb['n_total']} ({tie_rate:.1f}%) | "
                f"{pta_acc:.1f}% | {arm_acc:.1f}% | {tb['patch_helps']} | {tb['patch_hurts']} | "
                f"{b['image_proto']} ({100.0*b['image_proto']/npatch:.1f}%) | "
                f"{b['clip']} ({100.0*b['clip']/npatch:.1f}%) | "
                f"{b['other']} ({100.0*b['other']/npatch:.1f}%) | "
                f"{tb['gt_match']}/{npatch} ({gt_rate:.1f}%) |"
            )
    lines.append("")

    lines.append("## 4. flip_metrics_v2 Aggregate (pooled over 4 seeds)")
    lines.append("")
    lines.append("| Arm | Dataset | patch_alone corr/reg (net) | patch_alone acc | patch_image corr/reg (net) | patch_image acc |")
    lines.append("|---|---|---|---|---|---|")
    for arm in arms:
        for dataset in DATASETS:
            fa = flip_agg.get((arm, dataset))
            if not fa:
                lines.append(f"| {arm} | {dataset} | N/A | N/A | N/A | N/A |")
                continue
            pa, pi = fa["patch_alone"], fa["patch_image"]
            pa_acc = f"{pa['acc']:.1f}%" if pa.get("acc") is not None else "N/A"
            pi_acc = f"{pi['acc']:.1f}%" if pi.get("acc") is not None else "N/A"
            lines.append(
                f"| {arm} | {dataset} | {pa.get('corr',0)}/{pa.get('reg',0)} ({pa.get('net',0):+d}) | {pa_acc} | "
                f"{pi.get('corr',0)}/{pi.get('reg',0)} ({pi.get('net',0):+d}) | {pi_acc} |"
            )
    lines.append("")

    lines.append("## 5. Verdict")
    lines.append("")
    lines.append(f"**Overall study verdict: {overall_verdict}**")
    lines.append("")
    lines.append("| Arm | " + " | ".join(DATASETS) + " |")
    lines.append("|---|" + "---|" * len(DATASETS))
    for arm in arms:
        cells = [cell_verdicts.get((arm, d), "?") for d in DATASETS]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines.append("")

    return "\n".join(lines)


def build_verdict_json(deltas_stats, cell_verdicts, overall_verdict, thresholds, arms):
    cells = []
    for arm in arms:
        for dataset in DATASETS:
            ds = deltas_stats.get((arm, dataset), {})
            cells.append({
                "arm": arm, "dataset": dataset,
                "deltas": ds.get("deltas"),
                "delta_mean": ds.get("delta_mean"),
                "p_sign": ds.get("p_sign"),
                "seed_ci": ds.get("seed_ci"),
                "threshold": thresholds.get(dataset),
                "verdict": cell_verdicts.get((arm, dataset)),
            })
    return {"overall_verdict": overall_verdict, "cells": cells}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", required=True)
    p.add_argument("--prereg", default="outputs/patch_benefit_prereg.md")
    p.add_argument("--out", default="outputs/patch_benefit_report.md")
    p.add_argument("--verdict-json", default="outputs/patch_benefit_verdict.json")
    p.add_argument("--lock-noise-floor", action="store_true")
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args(argv)

    if args.lock_noise_floor:
        lock_noise_floor(args.records, args.prereg)
        return 0

    index = load_all(args.records)
    all_arms = [BASELINE_ARM] + PATCH_ARMS
    problems = verify(index, all_arms)
    hard_fails = [x for x in problems if not x.startswith("WARN")]
    for x in problems:
        print(x, file=sys.stderr)
    if hard_fails:
        print(f"\n{len(hard_fails)} hard verification failure(s).", file=sys.stderr)
        if args.verify_only:
            return 1
        raise SystemExit(f"Refusing to compute report with {len(hard_fails)} missing/bad cells.")
    if args.verify_only:
        print("[OK] verification passed (see WARNs above, if any).")
        return 0

    thresholds = read_locked_thresholds(args.prereg)

    acc_table = compute_accuracy_table(index, all_arms)
    deltas_stats = compute_deltas_and_stats(index, PATCH_ARMS)
    tie_breakdown = compute_tie_breakdown(index, PATCH_ARMS)
    flip_agg = compute_flip_aggregate(index, PATCH_ARMS)
    cell_verdicts, overall_verdict = apply_verdict(deltas_stats, flip_agg, thresholds, PATCH_ARMS)

    report = render_report(
        acc_table, deltas_stats, tie_breakdown, flip_agg,
        cell_verdicts, overall_verdict, thresholds, PATCH_ARMS,
    )
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"[OK] report written to {args.out}")

    verdict_json = build_verdict_json(deltas_stats, cell_verdicts, overall_verdict, thresholds, PATCH_ARMS)
    Path(args.verdict_json).write_text(json.dumps(verdict_json, indent=2), encoding="utf-8")
    print(f"[OK] verdict JSON written to {args.verdict_json}")
    print(f"\nOVERALL VERDICT: {overall_verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
