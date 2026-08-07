#!/usr/bin/env python3
"""
analyze_records.py — table / parity / select / hypothesis subcommands.

Consumes the per-sample JSONL records produced by ``utils.records.py``
(schema fixed in ``.omo/plans/perclass-difficult-classes-validation.md``, T1).
Record dir layout:

    DIR/<LABEL>/records.jsonl     per-sample stream (header line + sample lines)
    DIR/<LABEL>/summary.json      optional per-class summary (NOT required —
                                  per-class stats are recomputed from JSONL)

Label families (see ``classify_label``):
    PatchModPTA-CS-s1 / -s2            -> PatchModPTA   (main)
    PTA-CS-s1 / -s2                    -> PTA           (main)
    ZeroShot-CS-s1 / -s2               -> ZeroShot      (main)
    PatchModPTA-CS-alpha10-s1 /
    PatchModPTA-CS-multigate-s1        -> ablations (excluded from main analysis)

Subcommands:
    table      --records DIR [--out FILE]
    parity     --records DIR [--exp-results FILE]
    select     --records DIR [--k K] [--out FILE]
    hypothesis --records DIR [--win W] [--out FILE]

The hypothesis decision rule is PRE-REGISTERED — do not change:
    SUPPORT     if delta_mean > +1.0 pp AND rho_mean > 0.2
    REFUTE      if delta_mean < 0
    INCONCLUSIVE otherwise
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Pre-registered constants (fixed by the plan)
# ---------------------------------------------------------------------------

PATCH_PREFIX = "PatchModPTA-CS"
PTA_PREFIX = "PTA-CS"
ZERO_PREFIX = "ZeroShot-CS"
ABLATION_MARKERS = ("-alpha10", "-multigate")

METHOD_PATCH = "PatchModPTA"
METHOD_PTA = "PTA"
METHOD_ZERO = "ZeroShot"

PARITY_TOL = 0.5          # +/- percentage points around the published value
DECISION_DELTA_PP = 1.0   # SUPPORT needs delta_mean > +1.0 pp
DECISION_RHO = 0.2        # SUPPORT needs rho_mean > 0.2

DEFAULT_EXP_RESULTS = "outputs/exp_results.txt"
DEFAULT_TABLE_OUT = "outputs/perclass_tables.md"
DEFAULT_SELECT_OUT = "outputs/subset_classes.txt"
DEFAULT_HYPOTHESIS_OUT = "outputs/hypothesis_metrics.md"


# ---------------------------------------------------------------------------
# Record loading
# ---------------------------------------------------------------------------

def list_labels(records_dir):
    """Sorted labels = subdirs of ``records_dir`` containing ``records.jsonl``."""
    d = Path(records_dir)
    if not d.is_dir():
        return []
    labels = []
    for sub in sorted(d.iterdir()):
        if sub.is_dir() and (sub / "records.jsonl").is_file():
            labels.append(sub.name)
    return labels


def load_label_records(records_dir, label):
    """Return ``(header, samples)`` for ``DIR/label/records.jsonl``.

    ``header`` is the ``{"__header__": true, ...}`` line (or None if absent);
    ``samples`` is the list of per-sample dicts.
    """
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


def header_classnames(header):
    cn = (header or {}).get("classnames") or []
    return [str(c) for c in cn]


def dataset_classnames_fallback(header):
    """Best-effort classnames from the datasets module (lazy, defensive)."""
    try:
        dataset = (header or {}).get("dataset") or "dtd"
        from datasets import build_dataset
        d = build_dataset(dataset, "./data")
        cn = list(getattr(d, "classnames", []))
        return [str(c) for c in cn] if cn else []
    except Exception:
        return []


def resolve_classnames(records_dir, labels, headers):
    """Dataset-ordered classnames: first non-empty header list, else lookup."""
    del records_dir, labels  # kept for signature stability / future use
    for h in headers:
        cn = header_classnames(h)
        if cn:
            return cn
    for h in headers:
        cn = dataset_classnames_fallback(h)
        if cn:
            return cn
    return None


# ---------------------------------------------------------------------------
# Label classification / seed
# ---------------------------------------------------------------------------

def classify_label(label):
    """Map a record-dir label to a method family; None for ablations/unknown."""
    if not label:
        return None
    if any(m in label for m in ABLATION_MARKERS):
        return None
    if label.startswith(PATCH_PREFIX):
        return METHOD_PATCH
    if label.startswith(PTA_PREFIX):
        return METHOD_PTA
    if label.startswith(ZERO_PREFIX):
        return METHOD_ZERO
    return None


def is_ablation(label):
    return any(m in label for m in ABLATION_MARKERS)


def label_seed(label, header):
    """Run seed: header ``seed`` field, else ``-s<N>`` label suffix, else None."""
    h = header or {}
    if h.get("seed") is not None:
        try:
            return int(h["seed"])
        except (TypeError, ValueError):
            pass
    m = re.search(r"-s(\d+)(?:-|$)", label)
    if m:
        return int(m.group(1))
    return None


def find_seed_records(records_dir, method, seed):
    """Return ``(label, samples)`` for the first main ``method`` label at seed.

    Falls back to the first main label of the method when no seed matches.
    """
    best = (None, None)
    for label in list_labels(records_dir):
        if classify_label(label) != method:
            continue
        h, samples = load_label_records(records_dir, label)
        if label_seed(label, h) == seed:
            return label, samples
        if best[0] is None and samples:
            best = (label, samples)
    return best


# ---------------------------------------------------------------------------
# Per-class statistics (recomputed from JSONL target/correct)
# ---------------------------------------------------------------------------

def per_class_stats(samples):
    """``{target_idx: {"total": int, "correct": int}}`` recomputed from JSONL."""
    stats = {}
    for s in samples:
        t = int(s.get("target", -1))
        st = stats.setdefault(t, {"total": 0, "correct": 0})
        st["total"] += 1
        if s.get("correct"):
            st["correct"] += 1
    return stats


def overall_acc(samples):
    """Overall stream accuracy as a percentage float (None if empty)."""
    if not samples:
        return None
    return 100.0 * sum(1 for s in samples if s.get("correct")) / len(samples)


def per_class_named(samples, classnames):
    """``{classname: {"total", "correct", "acc"}}`` in dataset-name order."""
    stats = per_class_stats(samples)
    out = {}
    for t, st in stats.items():
        name = classnames[t] if classnames and t < len(classnames) else "class_{}".format(t)
        out[name] = {
            "total": st["total"],
            "correct": st["correct"],
            "acc": 100.0 * st["correct"] / st["total"] if st["total"] else 0.0,
        }
    return out


def _class_means(by_label):
    """``{classname: mean acc across labels}`` from ``{label: {name: st}}``."""
    accs = defaultdict(list)
    for named in by_label.values():
        for name, st in named.items():
            accs[name].append(st["acc"])
    return {name: sum(v) / len(v) for name, v in accs.items()}


def _name_sort_key(name):
    m = re.match(r"^class_(\d+)$", name)
    return (0, int(m.group(1))) if m else (1, name)


def _ordered_classnames(classnames, named):
    """Order ``named`` keys: dataset order first, then any leftovers."""
    ordered = []
    seen = set()
    if classnames:
        for name in classnames:
            if name in named:
                ordered.append(name)
                seen.add(name)
    for name in sorted(named, key=_name_sort_key):
        if name not in seen:
            ordered.append(name)
    return ordered


def _fmt(v):
    return "{:.2f}".format(v) if v is not None else "—"


def _fmt_delta(v):
    if v is None:
        return "—"
    return "{:+.2f}".format(v)


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------

def cmd_table(args):
    records_dir = Path(args.records)
    out_path = Path(args.out)
    labels = list_labels(records_dir)
    if not labels:
        print("[ERROR] no records found in {}".format(records_dir), file=sys.stderr)
        return 1

    headers = [load_label_records(records_dir, label)[0] for label in labels]
    classnames = resolve_classnames(records_dir, labels, headers)

    lines = ["# Per-Class Accuracy Tables", ""]
    lines.append("Generated from `{}`".format(records_dir))
    lines.append("")
    lines.append("## Per-method / per-seed per-class accuracy")
    lines.append("")
    lines.append("| method | seed | class | n | correct | acc (%) |")
    lines.append("|---|---|---|---|---|---|")

    by_method = defaultdict(dict)  # method -> label -> {classname: st}
    for label in labels:
        method = classify_label(label)
        if method is None:
            continue
        h, samples = load_label_records(records_dir, label)
        if not samples:
            continue
        named = per_class_named(samples, classnames)
        by_method[method][label] = named
        seed = label_seed(label, h)
        seed_txt = seed if seed is not None else ""
        for name in _ordered_classnames(classnames, named):
            st = named[name]
            lines.append("| {} | {} | {} | {} | {} | {:.2f} |".format(
                method, seed_txt, name, st["total"], st["correct"], st["acc"]))
    lines.append("")

    lines.append("## Cross-method per-class diff (mean across seeds)")
    lines.append("")
    lines.append("| class | PatchModPTA | PTA | PatchModPTA-PTA | ZeroShot | PatchModPTA-ZeroShot |")
    lines.append("|---|---|---|---|---|---|")
    method_means = {m: _class_means(by_method[m]) for m in by_method}
    all_classes = set()
    for means in method_means.values():
        all_classes.update(means)
    for name in _ordered_classnames(classnames, {c: {} for c in all_classes}):
        p = method_means.get(METHOD_PATCH, {}).get(name)
        q = method_means.get(METHOD_PTA, {}).get(name)
        z = method_means.get(METHOD_ZERO, {}).get(name)
        dp = (p - q) if (p is not None and q is not None) else None
        dz = (p - z) if (p is not None and z is not None) else None
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            name, _fmt(p), _fmt(q), _fmt_delta(dp), _fmt(z), _fmt_delta(dz)))
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("[OK] table written to {}".format(out_path))
    return 0


# ---------------------------------------------------------------------------
# parity
# ---------------------------------------------------------------------------

def load_reference_dtd(exp_results_path):
    """Parse ``exp_results.txt`` into ``{method_label: dtd_acc}``.

    Format: whitespace-separated fixed-width table; the FIRST numeric column
    is the ``dtd`` accuracy.  Header line is skipped (second token not a
    number).  Compares values, never labels.
    """
    refs = {}
    path = Path(exp_results_path)
    if not path.is_file():
        return refs
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            dtd = float(parts[1])
        except ValueError:
            continue  # header or non-numeric line
        refs[parts[0]] = dtd
    return refs


def find_reference(refs, method_prefix, preferred=None):
    """``(label, dtd)`` reference for a method family; preferred label first."""
    if preferred and preferred in refs:
        return preferred, refs[preferred]
    for label, v in refs.items():
        if label.startswith(method_prefix):
            return label, v
    return None, None


def cmd_parity(args):
    refs = load_reference_dtd(args.exp_results)
    patch_ref_label, patch_ref = find_reference(refs, PATCH_PREFIX, preferred="PatchModPTA-CS-2")
    pta_ref_label, pta_ref = find_reference(refs, PTA_PREFIX, preferred="PTA-CS")

    patch_label, patch_samples = find_seed_records(args.records, METHOD_PATCH, 1)
    pta_label, pta_samples = find_seed_records(args.records, METHOD_PTA, 1)

    patch_acc = overall_acc(patch_samples)
    pta_acc = overall_acc(pta_samples)

    def close(v, ref):
        return v is not None and ref is not None and abs(v - ref) <= PARITY_TOL

    ok = bool(close(patch_acc, patch_ref) and close(pta_acc, pta_ref))

    def txt(v):
        return "{:.2f}".format(v) if v is not None else "MISSING"

    verdict = "PASS" if ok else "FAIL"
    print(
        "PARITY GATE: {} ({} acc={} vs ref {} [{}]; {} acc={} vs ref {} [{}])".format(
            verdict,
            patch_label or "PatchModPTA-CS",
            txt(patch_acc),
            txt(patch_ref),
            patch_ref_label or "?",
            pta_label or "PTA-CS",
            txt(pta_acc),
            txt(pta_ref),
            pta_ref_label or "?",
        )
    )
    if not ok:
        print(
            "[ERROR] parity gate FAILED — records do not reproduce published "
            "results; do not trust further analysis.",
            file=sys.stderr,
        )
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------

def cmd_select(args):
    records_dir = Path(args.records)
    k = max(int(args.k), 1)
    out_path = Path(args.out)
    labels = list_labels(records_dir)
    if not labels:
        print("[ERROR] no records found in {}".format(records_dir), file=sys.stderr)
        return 1

    headers = [load_label_records(records_dir, label)[0] for label in labels]
    classnames = resolve_classnames(records_dir, labels, headers)

    # Mean PatchModPTA per-class accuracy across seeds (ablations excluded).
    patch_named = []
    for label in labels:
        if classify_label(label) != METHOD_PATCH:
            continue
        _, samples = load_label_records(records_dir, label)
        if samples:
            patch_named.append(per_class_named(samples, classnames))
    if not patch_named:
        print("[ERROR] no PatchModPTA-CS records found", file=sys.stderr)
        return 1

    mean_acc = {}
    for named in patch_named:
        for name, st in named.items():
            mean_acc.setdefault(name, []).append(st["acc"])
    mean_acc = {name: sum(v) / len(v) for name, v in mean_acc.items()}

    ranked = sorted(mean_acc.items(), key=lambda kv: (kv[1], _name_sort_key(kv[0])))
    selected = [name for name, _ in ranked[:k]]

    # Write in dataset.classnames order.
    if classnames:
        order = {name: i for i, name in enumerate(classnames)}
        selected = sorted(selected, key=lambda name: order.get(name, len(classnames)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")

    # Gap table for the selected classes.
    method_means = {}
    for method in (METHOD_PATCH, METHOD_PTA, METHOD_ZERO):
        accs = defaultdict(list)
        for label in labels:
            if classify_label(label) != method:
                continue
            _, samples = load_label_records(records_dir, label)
            if samples:
                for name, st in per_class_named(samples, classnames).items():
                    accs[name].append(st["acc"])
        method_means[method] = {name: sum(v) / len(v) for name, v in accs.items()}

    print("Selected {} difficult class(es) (bottom-{} by mean PatchModPTA-CS acc):".format(len(selected), k))
    print("| class | PatchModPTA | PTA | PatchModPTA-PTA | ZeroShot | PatchModPTA-ZeroShot |")
    print("|---|---|---|---|---|---|")
    for name in selected:
        p = method_means[METHOD_PATCH].get(name)
        q = method_means[METHOD_PTA].get(name)
        z = method_means[METHOD_ZERO].get(name)
        dp = (p - q) if (p is not None and q is not None) else None
        dz = (p - z) if (p is not None and z is not None) else None
        print("| {} | {} | {} | {} | {} | {} |".format(
            name, _fmt(p), _fmt(q), _fmt_delta(dp), _fmt(z), _fmt_delta(dz)))
    print("[OK] subset classes written to {}".format(out_path))
    return 0


# ---------------------------------------------------------------------------
# hypothesis — pre-registered metrics + decision rule
# ---------------------------------------------------------------------------

def resolve_tau_img(cfg):
    """``tau_image_proto`` from the resolved config (top-level or nested)."""
    if not isinstance(cfg, dict):
        return None
    for key in ("tau_image_proto", "tau_img"):
        v = cfg.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    for sub in ("fusion", "patch_level", "image_level", "text_level"):
        v = cfg.get(sub)
        if isinstance(v, dict):
            found = resolve_tau_img(v)
            if found is not None:
                return found
    return None


def rolling_windows(samples, win):
    """Non-overlapping ``win``-sized windows (+ trailing partial): [(start, end, acc)]."""
    windows = []
    n = len(samples)
    if n == 0 or win <= 0:
        return windows
    for start in range(0, n, win):
        end = min(start + win, n)
        chunk = samples[start:end]
        acc = 100.0 * sum(1 for s in chunk if s.get("correct")) / len(chunk)
        windows.append((start, end, acc))
    return windows


def per_class_rho(samples):
    """Spearman(cluster-growth, last-half per-class accuracy), zero-cluster classes masked.

    Returns ``{"rho": float, "pairs": [(class_idx, n_clusters_end, last_half_acc), ...],
               "masked": [class_idx, ...]}``.
    """
    n = len(samples)
    half = n // 2
    last = samples[half:]

    last_clusters = {}
    for s in samples:
        t = int(s.get("target", -1))
        ps = (s.get("proto_stats") or {}).get("true")
        if isinstance(ps, dict) and ps.get("n_clusters") is not None:
            last_clusters[t] = int(ps["n_clusters"])

    last_acc = defaultdict(lambda: [0, 0])  # class -> [total, correct]
    for s in last:
        t = int(s.get("target", -1))
        last_acc[t][0] += 1
        if s.get("correct"):
            last_acc[t][1] += 1

    pairs = []
    masked = []
    for c in sorted(set(last_clusters) | set(last_acc)):
        if c not in last_clusters or c not in last_acc:
            continue
        ncl = last_clusters[c]
        total, corr = last_acc[c]
        acc = 100.0 * corr / total if total else 0.0
        if ncl > 0:
            pairs.append((c, ncl, acc))
        else:
            masked.append(c)

    if len(pairs) < 2:
        rho = 0.0
    else:
        rho = spearman_rho([p[1] for p in pairs], [p[2] for p in pairs])
    return {"rho": rho, "pairs": pairs, "masked": masked}


def _rankdata(a):
    """Average ranks for tied values (scipy.stats.rankdata convention)."""
    order = sorted(range(len(a)), key=lambda i: a[i])
    ranks = [0.0] * len(a)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman_rho(x, y):
    """Spearman rank correlation; scipy if available, else numpy rank-based fallback."""
    if len(x) < 2 or len(y) < 2:
        return 0.0
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(x, y)
        rho = float(rho)
        if rho == rho:  # not NaN
            return rho
    except Exception:
        pass
    try:
        import numpy as np
        rx, ry = _rankdata(x), _rankdata(y)
        with np.errstate(all="ignore"):  # constant input -> NaN, handled below
            v = float(np.corrcoef(rx, ry)[0, 1])
        if v == v:  # not NaN
            return v
    except Exception:
        pass
    return _pearson(_rankdata(x), _rankdata(y))


def _pearson(x, y):
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def argmax_idx(vec):
    if not vec:
        return None
    return max(range(len(vec)), key=lambda i: vec[i])


def flip_metrics(samples, tau_img=None):
    """Offline per-class flip diagnostics (NOT part of the decision rule).

    Returns ``{class_idx: {"n", "patch_flip", "flip_vs_img", "patch_only_flip"}}``
    with the latter two as percentages (None where the inputs are unavailable /
    masked).
    """
    per = {}
    for s in samples:
        t = int(s.get("target", -1))
        lg = s.get("logits") or {}
        clip = lg.get("clip")
        final = lg.get("final")
        img = lg.get("image_proto")
        patch = lg.get("patch_proto")
        if not clip or not final:
            continue
        e = per.setdefault(t, {
            "n": 0, "patch_flip": 0, "flip_vs_img": 0, "img_denom": 0,
            "patch_only_flip": 0, "patch_only_denom": 0,
        })
        e["n"] += 1
        a_clip = argmax_idx(clip)
        if a_clip is None:
            continue
        if argmax_idx(final) != a_clip:
            e["patch_flip"] += 1
        if img is not None and tau_img is not None and len(img) == len(clip):
            e["img_denom"] += 1
            img_final = [c + tau_img * g for c, g in zip(clip, img)]
            if argmax_idx(img_final) != a_clip:
                e["flip_vs_img"] += 1
        ps = (s.get("proto_stats") or {}).get("true") or {}
        n_clusters = ps.get("n_clusters")
        if n_clusters and patch is not None and len(patch) == len(clip):
            e["patch_only_denom"] += 1
            if argmax_idx(patch) != a_clip:
                e["patch_only_flip"] += 1

    out = {}
    for t, e in per.items():
        out[t] = {
            "n": e["n"],
            "patch_flip": 100.0 * e["patch_flip"] / e["n"] if e["n"] else 0.0,
            "flip_vs_img": 100.0 * e["flip_vs_img"] / e["img_denom"] if e["img_denom"] else None,
            "patch_only_flip": 100.0 * e["patch_only_flip"] / e["patch_only_denom"] if e["patch_only_denom"] else None,
        }
    return out


def decide(delta_mean, rho_mean):
    """Pre-registered decision rule. Returns SUPPORT | REFUTE | INCONCLUSIVE."""
    if delta_mean > DECISION_DELTA_PP and rho_mean > DECISION_RHO:
        return "SUPPORT"
    if delta_mean < 0.0:
        return "REFUTE"
    return "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# flip metrics v2 — ground-truth-aware corrections/regressions
# (additive on top of flip_metrics; NOT part of the decision rule)
# ---------------------------------------------------------------------------

def resolve_tau_patch_proto(cfg):
    """``tau_patch_proto`` from the resolved config (top-level or nested).

    Absent on PTA/ZeroShot paths -> None (callers treat it as 0: those
    mechanisms never add a patch term).
    """
    if not isinstance(cfg, dict):
        return None
    v = cfg.get("tau_patch_proto")
    if v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    for sub in ("fusion", "patch_level", "image_level", "text_level"):
        v = cfg.get(sub)
        if isinstance(v, dict):
            found = resolve_tau_patch_proto(v)
            if found is not None:
                return found
    return None


def _acc_v2(e):
    """Accuracy summary from a ``{"n", "correct"}`` counter dict."""
    return {
        "n": e["n"],
        "correct": e["correct"],
        "acc": 100.0 * e["correct"] / e["n"] if e["n"] else None,
    }


def _flip_v2(e):
    """Corrections/regressions summary from a corr/reg counter dict."""
    denom = e["corr"] + e["reg"]
    return {
        "n": e["n"],
        "corr": e["corr"],
        "reg": e["reg"],
        "acc": 100.0 * e["corr"] / denom if denom else None,
        "net": e["corr"] - e["reg"],
        "masked": e.get("masked", 0),
    }


def flip_metrics_v2(samples, tau_img=None, tau_patch_proto=None):
    """GT-aware per-component corrections/regressions (diagnostic, v2).

    Reconstructs each component prediction from STORED logits only — the
    ProtoAlphaFusion formula (fusion.py:123-138) with ``proto_alpha`` treated
    as 1.0 (the post-fix design; an old record's per-class alpha is
    deliberately NOT applied):

        text-only        = argmax(clip)
        image-only       = argmax(clip + tau_img * image_proto)
        patch-only       = argmax(patch_proto)
        patch-with-image = argmax(clip + tau_img*image_proto
                                  + tau_patch_proto*patch_proto)

    For each record with ``target`` known, counted relative to the previous
    component's prediction:

        correction  — previous component wrong, this component right
        regression  — previous component right, this component wrong
        flip acc    = corrections / (corrections + regressions)
        net         = corrections - regressions

    Patch rows are masked to records with ``proto_stats.pred.n_clusters > 0``
    (cluster availability; ``proto_stats.true`` is the fallback).  Classes
    with zero cluster-bearing records get no patch attribution — callers must
    mark them "patch contribution N/A".

    ``tau_img`` / ``tau_patch_proto`` come from the record header's
    ``resolved_config`` (see :func:`resolve_tau_img` /
    :func:`resolve_tau_patch_proto`).  ``tau_patch_proto=None`` disables the
    patch-with-image row (PTA/ZeroShot paths never add the patch term).

    Returns ``{"tau_img", "tau_patch_proto", "n", "aggregate", "per_class"}``;
    aggregate keys are ``text_only`` / ``image`` / ``patch_alone`` /
    ``patch_image`` and per_class maps class index to the same counters plus
    ``n_cluster`` (number of cluster-bearing records for that class).
    """
    agg = {
        "text_only": {"n": 0, "correct": 0},
        "image": {"n": 0, "corr": 0, "reg": 0},
        "patch_alone": {"n": 0, "corr": 0, "reg": 0, "masked": 0},
        "patch_image": {"n": 0, "corr": 0, "reg": 0, "masked": 0},
    }
    per = defaultdict(lambda: {
        "n": 0,
        "n_cluster": 0,
        "text_only": {"n": 0, "correct": 0},
        "image": {"n": 0, "corr": 0, "reg": 0},
        "patch_alone": {"n": 0, "corr": 0, "reg": 0},
        "patch_image": {"n": 0, "corr": 0, "reg": 0},
    })

    for s in samples:
        t = int(s.get("target", -1))
        lg = s.get("logits") or {}
        clip = lg.get("clip")
        if t < 0 or not clip:
            continue
        img = lg.get("image_proto")
        patch = lg.get("patch_proto")
        p_text = argmax_idx(clip)
        if p_text is None:
            continue

        e = per[t]
        e["n"] += 1
        text_correct = p_text == t
        agg["text_only"]["n"] += 1
        agg["text_only"]["correct"] += 1 if text_correct else 0
        e["text_only"]["n"] += 1
        e["text_only"]["correct"] += 1 if text_correct else 0

        ps = s.get("proto_stats") or {}
        ncl = None
        if isinstance(ps.get("pred"), dict) and ps["pred"].get("n_clusters") is not None:
            ncl = int(ps["pred"]["n_clusters"])
        elif isinstance(ps.get("true"), dict) and ps["true"].get("n_clusters") is not None:
            ncl = int(ps["true"]["n_clusters"])
        has_clusters = (ncl or 0) > 0
        if has_clusters:
            e["n_cluster"] += 1

        img_ok = img is not None and tau_img is not None and len(img) == len(clip)
        patch_ok = patch is not None and len(patch) == len(clip)

        # +image (PTA mechanism): clip + tau_img*image_proto vs text-only.
        p_img = None
        if img_ok:
            p_img = argmax_idx([c + tau_img * g for c, g in zip(clip, img)])
        if p_img is not None:
            agg["image"]["n"] += 1
            e["image"]["n"] += 1
            if p_img == t:
                if not text_correct:
                    agg["image"]["corr"] += 1
                    e["image"]["corr"] += 1
            elif text_correct:
                agg["image"]["reg"] += 1
                e["image"]["reg"] += 1

        # +patch alone: argmax(patch_proto) vs text-only (cluster-masked).
        if has_clusters and patch_ok:
            p_patch = argmax_idx(patch)
            if p_patch is not None:
                agg["patch_alone"]["n"] += 1
                e["patch_alone"]["n"] += 1
                if p_patch == t:
                    if not text_correct:
                        agg["patch_alone"]["corr"] += 1
                        e["patch_alone"]["corr"] += 1
                elif text_correct:
                    agg["patch_alone"]["reg"] += 1
                    e["patch_alone"]["reg"] += 1
        else:
            agg["patch_alone"]["masked"] += 1

        # +patch with image: + tau_patch_proto*patch_proto vs image-only
        # (isolates what patch ADDS on top of PTA; cluster-masked).
        if has_clusters and patch_ok and img_ok and tau_patch_proto is not None:
            p_patch_img = argmax_idx(
                [c + tau_img * g + tau_patch_proto * p
                 for c, g, p in zip(clip, img, patch)])
            if p_patch_img is not None and p_img is not None:
                agg["patch_image"]["n"] += 1
                e["patch_image"]["n"] += 1
                img_correct = p_img == t
                if p_patch_img == t:
                    if not img_correct:
                        agg["patch_image"]["corr"] += 1
                        e["patch_image"]["corr"] += 1
                elif img_correct:
                    agg["patch_image"]["reg"] += 1
                    e["patch_image"]["reg"] += 1
        else:
            agg["patch_image"]["masked"] += 1

    out = {
        "tau_img": tau_img,
        "tau_patch_proto": tau_patch_proto,
        "n": agg["text_only"]["n"],
        "aggregate": {
            "text_only": _acc_v2(agg["text_only"]),
            "image": _flip_v2(agg["image"]),
            "patch_alone": _flip_v2(agg["patch_alone"]),
            "patch_image": _flip_v2(agg["patch_image"]),
        },
        "per_class": {},
    }
    for t in sorted(per):
        e = per[t]
        out["per_class"][t] = {
            "n": e["n"],
            "n_cluster": e["n_cluster"],
            "text_only": _acc_v2(e["text_only"]),
            "image": _flip_v2(e["image"]),
            "patch_alone": _flip_v2(e["patch_alone"]),
            "patch_image": _flip_v2(e["patch_image"]),
        }
    return out


def _flip_cell(e):
    """'corr/reg (acc%, net)' cell for the markdown tables (— when no flips)."""
    if e.get("acc") is None:
        return "—"
    return "{}/{} ({:.2f}%, {:+.0f})".format(
        e["corr"], e["reg"], e["acc"], e["net"])


def _render_v2_label(lines, label, classnames, m):
    """Append one label's aggregate + per-class v2 tables to ``lines``."""
    tau_img = m["tau_img"]
    tau_pp = m["tau_patch_proto"]
    tau_img_txt = "{:.1f}".format(tau_img) if tau_img is not None else "n/a"
    tau_pp_txt = "{:.1f}".format(tau_pp) if tau_pp is not None else "n/a (0.0)"
    lines.append("### {}  (tau_img = {}, tau_patch_proto = {})".format(
        label, tau_img_txt, tau_pp_txt))
    lines.append("")
    lines.append("| component | n | corrections | regressions | flip acc (%) | net | masked |")
    lines.append("|---|---|---|---|---|---|---|")
    a = m["aggregate"]
    txt = a["text_only"]
    lines.append("| text-only (baseline) | {} | {} | — | {} | — | — |".format(
        txt["n"], txt["correct"], _fmt(txt["acc"])))
    for name, key in (("+image (PTA)", "image"),
                      ("+patch alone", "patch_alone"),
                      ("+patch with image", "patch_image")):
        e = a[key]
        net_txt = "{:+.0f}".format(e["net"]) if e["acc"] is not None else "—"
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            name, e["n"], e["corr"], e["reg"], _fmt(e["acc"]), net_txt, e["masked"]))
    lines.append("")
    lines.append("| class | n | n_cluster | text-only acc (%) | +image (corr/reg, acc%, net) | "
                 "+patch alone (corr/reg, acc%, net) | +patch with image (corr/reg, acc%, net) |")
    lines.append("|---|---|---|---|---|---|---|")
    for cls in sorted(m["per_class"]):
        e = m["per_class"][cls]
        name = _classname_of(classnames, cls)
        if e["n_cluster"] == 0:
            pa_cell = pi_cell = "patch contribution N/A"
        else:
            pa_cell, pi_cell = _flip_cell(e["patch_alone"]), _flip_cell(e["patch_image"])
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            name, e["n"], e["n_cluster"], _fmt(e["text_only"]["acc"]),
            _flip_cell(e["image"]), pa_cell, pi_cell))
    lines.append("")

    print("flips-v2 {}: text-only acc={} ({}/{}); +image corr={} reg={} acc={} net={:+.0f}; "
          "+patch-alone corr={} reg={} acc={} net={:+.0f} (masked {}); "
          "+patch+img corr={} reg={} acc={} net={:+.0f} (masked {})".format(
        label, _fmt(a["text_only"]["acc"]), a["text_only"]["correct"], a["text_only"]["n"],
        a["image"]["corr"], a["image"]["reg"], _fmt(a["image"]["acc"]), a["image"]["net"],
        a["patch_alone"]["corr"], a["patch_alone"]["reg"],
        _fmt(a["patch_alone"]["acc"]), a["patch_alone"]["net"], a["patch_alone"]["masked"],
        a["patch_image"]["corr"], a["patch_image"]["reg"],
        _fmt(a["patch_image"]["acc"]), a["patch_image"]["net"], a["patch_image"]["masked"]))


def cmd_flips_v2(args):
    records_dir = Path(args.records)
    out_path = Path(args.out)
    labels = list_labels(records_dir)
    if not labels:
        print("[ERROR] no records found in {}".format(records_dir), file=sys.stderr)
        return 1

    headers = [load_label_records(records_dir, label)[0] for label in labels]
    classnames = resolve_classnames(records_dir, labels, headers)

    lines = ["# Ground-Truth-Aware Flip Metrics v2", ""]
    lines.append("Generated from `{}`".format(records_dir))
    lines.append("")
    lines.append("Per-component predictions reconstructed from STORED logits "
                 "(ProtoAlphaFusion formula, proto_alpha treated as 1.0):")
    lines.append("")
    lines.append("- text-only        = argmax(clip)")
    lines.append("- image-only       = argmax(clip + tau_img*image_proto)")
    lines.append("- patch-only       = argmax(patch_proto)")
    lines.append("- patch-with-image = argmax(clip + tau_img*image_proto "
                 "+ tau_patch_proto*patch_proto)")
    lines.append("")
    lines.append("`correction` = previous component wrong -> this component right; "
                 "`regression` = previous component right -> this component wrong.")
    lines.append("flip accuracy = corrections / (corrections + regressions); "
                 "net = corrections - regressions.")
    lines.append("")
    lines.append("Patch rows are masked to records with "
                 "`proto_stats.pred.n_clusters > 0`; classes with no "
                 "cluster-bearing records are marked `patch contribution N/A`.")
    lines.append("")

    for label in labels:
        header, samples = load_label_records(records_dir, label)
        if not samples:
            continue
        cfg = (header or {}).get("resolved_config")
        tau_img = resolve_tau_img(cfg)
        tau_pp = resolve_tau_patch_proto(cfg)
        m = flip_metrics_v2(samples, tau_img, tau_pp)
        _render_v2_label(lines, label, classnames, m)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[OK] flip metrics v2 written to {}".format(out_path))
    return 0


def _classname_of(classnames, idx):
    if classnames and idx < len(classnames):
        return classnames[idx]
    return "class_{}".format(idx)


def _build_hypothesis_report(records_dir, win, run_rows, classnames,
                             delta_mean, rho_mean, decision, patch_seeds_info):
    lines = ["# Hypothesis Metrics — stream-time accuracy vs patch-prototype growth", ""]
    lines.append("Generated from `{}`".format(records_dir))
    lines.append("")
    lines.append("Hypothesis (pre-registered): *as you build better patch-level "
                 "prototypes over time, classification performance increases.*")
    lines.append("")
    lines.append("Decision rule (PatchModPTA-CS full-DTD records only, ablations excluded):")
    lines.append("")
    lines.append("- SUPPORT if `delta_mean > +1.0 pp` **AND** `rho_mean > 0.2`")
    lines.append("- REFUTE if `delta_mean < 0`")
    lines.append("- INCONCLUSIVE otherwise")
    lines.append("")
    lines.append("`acc_first` = mean(correct) over first 50% of stream; "
                 "`acc_last` = mean(correct) over last 50%; `delta = acc_last - acc_first` (pp).")
    lines.append("`rho` = Spearman(cluster-growth vs last-half per-class accuracy), "
                 "masked to classes with nonzero clusters at stream end.")
    lines.append("")
    lines.append("## Per-run stream metrics")
    lines.append("")
    lines.append("| label | method | seed | n | acc_first (%) | acc_last (%) | delta (pp) | rho |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in run_rows:
        seed = r["seed"] if r["seed"] is not None else "?"
        lines.append("| {} | {} | {} | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |".format(
            r["label"], r["method"], seed, r["n"],
            r["acc_first"], r["acc_last"], r["delta"], r["rho"]))
    lines.append("")
    lines.append("## Rolling-window accuracy")
    lines.append("")
    lines.append("Window size = {} samples (non-overlapping).".format(win))
    for r in run_rows:
        lines.append("")
        lines.append("### {}".format(r["label"]))
        lines.append("")
        lines.append("| samples | n | acc (%) |")
        lines.append("|---|---|---|")
        for start, end, acc in r["rolling"]:
            lines.append("| [{}:{}) | {} | {:.2f} |".format(start, end, end - start, acc))
    lines.append("")
    lines.append("## Per-class cluster-growth vs last-half accuracy (Spearman rho)")
    for r in run_rows:
        lines.append("")
        lines.append("### {}  (rho = {:.2f})".format(r["label"], r["rho"]))
        lines.append("")
        if r["rho_pairs"]:
            lines.append("| class | n_clusters_end | last-half acc (%) |")
            lines.append("|---|---|---|")
            for cls, ncl, acc in r["rho_pairs"]:
                lines.append("| {} | {} | {:.2f} |".format(
                    _classname_of(classnames, cls), ncl, acc))
        if r["rho_masked"]:
            masked_names = ", ".join(_classname_of(classnames, c) for c in r["rho_masked"])
            lines.append("")
            lines.append("*Masked (zero clusters at end): {}*".format(masked_names))
    lines.append("")
    lines.append("## Flip diagnostics (offline, diagnostic only — NOT part of the decision rule)")
    lines.append("")
    lines.append("- `patch_flip`: argmax(final) != argmax(clip)")
    lines.append("- `flip_vs_img`: argmax(final) != argmax(clip + tau_img*image_proto) "
                 "(— when image_proto / tau unavailable)")
    lines.append("- `patch_only_flip`: argmax(patch_proto) != argmax(clip), "
                 "masked when the class has zero prototypes")
    for r in run_rows:
        lines.append("")
        tau_txt = "{:.1f}".format(r["tau_img"]) if r["tau_img"] is not None else "n/a"
        lines.append("### {}  (tau_img = {})".format(r["label"], tau_txt))
        lines.append("")
        lines.append("| class | n | patch_flip (%) | flip_vs_img (%) | patch_only_flip (%) |")
        lines.append("|---|---|---|---|---|")
        for cls in sorted(r["flips"]):
            f = r["flips"][cls]
            lines.append("| {} | {} | {:.2f} | {} | {} |".format(
                _classname_of(classnames, cls), f["n"],
                f["patch_flip"], _fmt(f["flip_vs_img"]), _fmt(f["patch_only_flip"])))
    lines.append("")
    lines.append("## Aggregate over PatchModPTA-CS seeds")
    lines.append("")
    lines.append("- delta_mean = {:.2f} pp".format(delta_mean))
    lines.append("- rho_mean = {:.2f}".format(rho_mean))
    if patch_seeds_info:
        lines.append("")
        lines.append("Per-seed:")
        for label, d, rho in patch_seeds_info:
            lines.append("  - {}: delta = {:.2f} pp, rho = {:.2f}".format(label, d, rho))
    lines.append("")
    lines.append("DECISION: {} (delta_mean={:.2f} pp, rho_mean={:.2f})".format(
        decision, delta_mean, rho_mean))
    return lines


def cmd_hypothesis(args):
    records_dir = Path(args.records)
    out_path = Path(args.out)
    win = max(int(args.win), 1)

    labels = list_labels(records_dir)
    if not labels:
        print("[ERROR] no records found in {}".format(records_dir), file=sys.stderr)
        return 1

    headers = [load_label_records(records_dir, label)[0] for label in labels]
    classnames = resolve_classnames(records_dir, labels, headers)

    run_rows = []
    patch_deltas, patch_rhos, patch_seeds_info = [], [], []

    for label in labels:
        method = classify_label(label)
        if method is None:
            continue
        header, samples = load_label_records(records_dir, label)
        if not samples:
            continue
        seed = label_seed(label, header)
        n = len(samples)
        half = n // 2
        first, last = samples[:half], samples[half:]
        acc_first = 100.0 * sum(1 for s in first if s.get("correct")) / len(first) if first else 0.0
        acc_last = 100.0 * sum(1 for s in last if s.get("correct")) / len(last) if last else 0.0
        delta = acc_last - acc_first

        rho_info = per_class_rho(samples)
        tau_img = resolve_tau_img(header.get("resolved_config"))
        run = {
            "label": label, "method": method, "seed": seed, "n": n,
            "acc_first": acc_first, "acc_last": acc_last, "delta": delta,
            "rho": rho_info["rho"], "rho_pairs": rho_info["pairs"],
            "rho_masked": rho_info["masked"],
            "flips": flip_metrics(samples, tau_img),
            "rolling": rolling_windows(samples, win),
            "tau_img": tau_img,
        }
        run_rows.append(run)
        if method == METHOD_PATCH:
            patch_deltas.append(delta)
            patch_rhos.append(rho_info["rho"])
            patch_seeds_info.append((label, delta, rho_info["rho"]))

    delta_mean = sum(patch_deltas) / len(patch_deltas) if patch_deltas else 0.0
    rho_mean = sum(patch_rhos) / len(patch_rhos) if patch_rhos else 0.0
    decision = decide(delta_mean, rho_mean)

    lines = _build_hypothesis_report(records_dir, win, run_rows, classnames,
                                     delta_mean, rho_mean, decision, patch_seeds_info)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("DECISION: {} (delta_mean={:.2f} pp, rho_mean={:.2f})".format(
        decision, delta_mean, rho_mean))
    print("[OK] hypothesis metrics written to {}".format(out_path))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Analyze per-sample JSONL records (table / parity / select / hypothesis).")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_table = sub.add_parser("table", help="per-class accuracy tables + cross-method diff")
    p_table.add_argument("--records", required=True, help="record dir DIR/<LABEL>/records.jsonl")
    p_table.add_argument("--out", default=DEFAULT_TABLE_OUT)
    p_table.set_defaults(func=cmd_table)

    p_parity = sub.add_parser("parity", help="parity gate vs published results (exit 1 on FAIL)")
    p_parity.add_argument("--records", required=True, help="record dir DIR/<LABEL>/records.jsonl")
    p_parity.add_argument("--exp-results", default=DEFAULT_EXP_RESULTS,
                          help="reference results file (default: outputs/exp_results.txt)")
    p_parity.set_defaults(func=cmd_parity)

    p_select = sub.add_parser("select", help="bottom-k difficult classes by mean PatchModPTA acc")
    p_select.add_argument("--records", required=True, help="record dir DIR/<LABEL>/records.jsonl")
    p_select.add_argument("--k", type=int, default=5)
    p_select.add_argument("--out", default=DEFAULT_SELECT_OUT)
    p_select.set_defaults(func=cmd_select)

    p_hyp = sub.add_parser("hypothesis", help="pre-registered hypothesis metrics + decision rule")
    p_hyp.add_argument("--records", required=True, help="record dir DIR/<LABEL>/records.jsonl")
    p_hyp.add_argument("--win", type=int, default=100, help="rolling-window size (default 100)")
    p_hyp.add_argument("--out", default=DEFAULT_HYPOTHESIS_OUT)
    p_hyp.set_defaults(func=cmd_hypothesis)

    p_flips2 = sub.add_parser(
        "flips-v2",
        help="GT-aware flip metrics v2 (corrections/regressions per component)")
    p_flips2.add_argument("--records", required=True,
                          help="record dir DIR/<LABEL>/records.jsonl")
    p_flips2.add_argument("--out", default="outputs/flip_metrics_v2.md",
                          help="markdown output path")
    p_flips2.set_defaults(func=cmd_flips_v2)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
