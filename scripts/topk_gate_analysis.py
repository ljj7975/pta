#!/usr/bin/env python3
"""Write-gate comparison on FULL DTD, offline from records.

The write-side gate decides WHICH class memories get updated per image.
Current rule (patch_modulated_pta.py:249): update ALL classes c with
softmax(clip_logits)[c] >= 0.1.

This script compares candidate rules, including CLASS-COUNT-INDEPENDENT
ones. Fixed thresholds/margins (e.g. band M=0.5, floor=0.05) are
calibrated to DTD's 47 classes: softmax magnitudes shrink as the number
of classes grows, so the same M/floor means something different on a
5-way subset or 1000-way ImageNet. The generic rules below are expressed
in units that do not depend on C:

  ratio      w_c >= r * w_top1            (r = temperature-scaled logit gap:
                                           w_c/w_1 = exp((l_c-l_1)/T))
  cpm        write top classes until      (P = fraction of the model's total
             cumulative mass reaches P     belief; adaptive rank depth)
  kuniform   w_c >= k/C (k times uniform  (floor in fair-share units: a class
             share) AND w_c >= w_1 -       must hold k/C of the mass and be
             m*(1-1/C)                     within a fraction m of the [1/C,1]
                                           probability range)

A simulation then re-runs every rule on random C-class subsamples of the
same logits (true class kept, softmax renormalized) to show which rules
drift as C changes and which stay stable.
"""
import json
import math
import os
import random

RECORDS_DIR = "outputs/records"
TAU_IMAGE = 80.0  # resolved config: fusion.tau_image_proto
T = 20.0          # image_level EMA temperature (w_new = 1 - exp(-w/T))


def softmax(v):
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]


def load(label, seed):
    out = []
    with open(os.path.join(RECORDS_DIR, f"{label}-s{seed}", "records.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("__header__"):
                continue
            out.append(rec)
    return out


def logits_of(rec, source):
    if source == "clip":
        return rec["logits"]["clip"]
    if source == "image_proto":
        return rec["logits"]["image_proto"]
    if source == "patch_proto":
        return rec["logits"]["patch_proto"]
    if source == "final":
        return rec["logits"]["final"]
    if source == "pta":
        clip = rec["logits"]["clip"]
        img = rec["logits"]["image_proto"]
        return [c + TAU_IMAGE * i for c, i in zip(clip, img)]
    raise ValueError(source)


# --- selectors: each takes (w, t) where w = softmax probs, t = true class ---
def sel_thresh(w, t, thr=0.1):
    return [c for c in range(len(w)) if w[c] >= thr]


def sel_topk(w, t, k=1):
    return sorted(range(len(w)), key=lambda i: w[i], reverse=True)[:k]


def sel_margin(w, t, margin=0.3):
    order = sorted(range(len(w)), key=lambda i: w[i], reverse=True)
    sel = [order[0]]
    if len(order) > 1 and w[order[0]] - w[order[1]] < margin:
        sel.append(order[1])
    return sel


def sel_band(w, t, M=0.5, floor=0.0):
    order = sorted(range(len(w)), key=lambda i: w[i], reverse=True)
    top1 = w[order[0]]
    return [c for c in order if w[c] >= top1 - M and w[c] >= floor]


def sel_ratio(w, t, r=0.5):
    """Class-count-independent: w_c >= r*w_top1 is a statement about the
    logit gap (exp((l_c-l_1)/T) >= r), not about the number of classes."""
    order = sorted(range(len(w)), key=lambda i: w[i], reverse=True)
    return [c for c in order if w[c] >= r * w[order[0]]]


def sel_cpm(w, t, P=0.5):
    """Cumulative-probability-mass: write the top classes until their mass
    reaches P. P is a statement about belief share, so it means the same
    thing at any class count; the depth adapts (confident -> 1 class)."""
    order = sorted(range(len(w)), key=lambda i: w[i], reverse=True)
    sel = []
    acc = 0.0
    for c in order:
        sel.append(c)
        acc += w[c]
        if acc >= P:
            break
    return sel


def sel_kuniform(w, t, k=2.0, m=0.5):
    """Fair-share rule: class c is written iff it holds k/C of the mass
    (k times its uniform share) AND stays within a fraction m of top-1
    measured across the non-uniform probability range [1/C, 1]. Both
    knobs are dimensionless fractions of the C-dependent range, so they
    transfer across datasets."""
    C = len(w)
    u = 1.0 / C
    order = sorted(range(len(w)), key=lambda i: w[i], reverse=True)
    top1 = w[order[0]]
    band = top1 - m * (1.0 - u)
    floor = k * u
    return [c for c in order if w[c] >= band and w[c] >= floor]


def run_rule(recs, source, select):
    """Evaluate any selector over the record stream. Returns
    (events/sample, true-coverage%, precision%, p90-max-rank, pos_mass,
    neg_mass). Weight-aware masses use w_new = 1-exp(-w/T) because deep-
    rank writes carry tiny weights, so equal-weight precision overstates
    their contamination. p90 rank = 90th percentile of the deepest rank
    written (1-based)."""
    n = len(recs)
    events = 0
    true_covered = 0
    true_events = 0
    pos_mass = 0.0
    neg_mass = 0.0
    sizes = []
    for rec in recs:
        logits = logits_of(rec, source)
        t = int(rec["target"])
        w = softmax(logits)
        sel = select(w, t)
        sizes.append(len(sel))
        for c in sel:
            w_new = 1.0 - math.exp(-w[c] / T)
            if c == t:
                pos_mass += w_new
                true_events += 1
            else:
                neg_mass += w_new
        events += len(sel)
        if t in sel:
            true_covered += 1
    sizes.sort()
    p90 = sizes[min(len(sizes) - 1, int(0.90 * len(sizes)))]
    return (events / n, 100.0 * true_covered / n,
            100.0 * true_events / events if events else 0.0,
            p90, pos_mass, neg_mass)


def simulate_C(recs, source, rules, Cs=(5, 10, 20, 47), seeds=(0, 1, 2)):
    """Re-run each rule on random C-class subsamples of the SAME logits
    (true class always kept, softmax renormalized over the subset), to see
    how rule behavior changes with the number of classes. Averages over
    seeds. Returns dict rule -> C -> (events/sample, cov%, prec%)."""
    per = [(logits_of(rec, source), int(rec["target"])) for rec in recs]
    acc = {name: {C: [0.0, 0.0, 0.0] for C in Cs} for name, _ in rules}
    for seed in seeds:
        rng = random.Random(seed)
        for lg, t in per:
            nC = len(lg)
            others = [c for c in range(nC) if c != t]
            for C in Cs:
                if C >= nC:
                    sub = list(range(nC))
                else:
                    sub = rng.sample(others, C - 1) + [t]
                subw = softmax([lg[c] for c in sub])
                lt = sub.index(t)
                for name, select in rules:
                    sel = select(subw, lt)
                    writes = len(sel)
                    twrites = sum(1 for c in sel if c == lt)
                    acc[name][C][0] += writes
                    acc[name][C][1] += 1.0 if lt in sel else 0.0
                    acc[name][C][2] += twrites
    denom = len(per) * len(seeds)
    out = {}
    for name in acc:
        out[name] = {}
        for C in Cs:
            ev, cov, te = acc[name][C]
            out[name][C] = (ev / denom, 100.0 * cov / denom,
                            100.0 * te / ev if ev else 0.0)
    return out


def fmt(ev, cov, prec):
    return f"{ev:>9.2f} {cov:>9.1f} {prec:>10.1f}"


def main():
    recs = load("PatchModPTA-CS", 1) + load("PatchModPTA-CS", 2)
    n = len(recs)
    print(f"records: {n} samples (2 seeds, full DTD, 47-way open-set)")
    print()

    header = f"{'rule':<26} {'source':<10} {'ev/sample':>9} {'true-cov%':>9} {'precision%':>10} {'p90 rank':>8}"
    print(header)
    print("-" * len(header))

    def row(name, source, select):
        ev, cov, prec, p90, _, _ = run_rule(recs, source, select)
        print(f"{name:<26} {source:<10} {ev:>9.2f} {cov:>9.1f} {prec:>10.1f} {p90:>8}")

    # --- baseline: current rule + fixed rules (class-count-sensitive) ---
    row("thresh>=0.1 (current)", "clip", sel_thresh)
    for k in (1, 2, 3):
        row(f"top-{k}", "final", lambda w, t, k=k: sel_topk(w, t, k))
    for m in (0.05, 0.3):
        row(f"margin<{m}", "final", lambda w, t, m=m: sel_margin(w, t, m))
    row("band M<0.5 floor>0.05", "final", lambda w, t: sel_band(w, t, 0.5, 0.05))
    print()

    # --- generic (class-count-independent) rules on final logits ---
    print("-- generic rules, source=final (class-count-independent units) --")
    print(header)
    print("-" * len(header))
    for r in (0.3, 0.5, 0.7):
        row(f"ratio r<{r}", "final", lambda w, t, r=r: sel_ratio(w, t, r))
    for P in (0.3, 0.5, 0.8):
        row(f"cpm P<{P}", "final", lambda w, t, P=P: sel_cpm(w, t, P))
    for k in (2.0, 3.0):
        for m in (0.3, 0.5):
            row(f"kuniform k<{k} m<{m}", "final",
                lambda w, t, k=k, m=m: sel_kuniform(w, t, k, m))
    print()

    # --- class-count sensitivity: same rules on subsampled class sets ---
    print("-- class-count simulation: rules re-run on C-class subsamples --")
    print("  (true class kept, softmax renormalized over the C classes;")
    print("   averaged over 3 random subsamples; clip=0.1 uses clip source,")
    print("   all others use final. 'current' should drift if C-sensitive.)")
    print()
    rules = [
        ("thresh>=0.1 (current)", sel_thresh),
        ("margin<0.3", lambda w, t: sel_margin(w, t, 0.3)),
        ("band M<0.5 fl>0.05", lambda w, t: sel_band(w, t, 0.5, 0.05)),
        ("ratio r<0.5", lambda w, t: sel_ratio(w, t, 0.5)),
        ("cpm P<0.5", lambda w, t: sel_cpm(w, t, 0.5)),
        ("kuniform k<2 m<0.5", lambda w, t: sel_kuniform(w, t, 2.0, 0.5)),
        ("kuniform k<3 m<0.5", lambda w, t: sel_kuniform(w, t, 3.0, 0.5)),
    ]
    sim_src = {"thresh>=0.1 (current)": "clip"}
    res = {}
    for name, select in rules:
        res[name] = simulate_C(recs, sim_src.get(name, "final"), [(name, select)])[name]
    Cs = (5, 10, 20, 47)
    hdr = f"{'rule':<26} " + " ".join(f"{'C=' + str(C):>32}" for C in Cs)
    print(hdr)
    print("-" * len(hdr))
    for name, _ in rules:
        cells = []
        for C in Cs:
            ev, cov, prec = res[name][C]
            cells.append(f"({ev:>5.2f},{cov:>5.1f}%,{prec:>5.1f}%)")
        print(f"{name:<26} " + " ".join(f"{c:>32}" for c in cells))
    print()
    print("cell format: (events/sample, true-coverage%, precision%)")


if __name__ == "__main__":
    main()
