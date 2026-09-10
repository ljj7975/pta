"""Aggregate representation-upgrade wave results.

Parses outputs_repr_upgrade/result_*.txt (per-run lines) + records_*/convergence.json
and prints 4-seed-mean tables vs the archived PTA / CLIP references, plus
convergence (AUC / early-fraction) and per-seed spreads.

Usage: python scripts/aggregate_repr_upgrade.py
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs_repr_upgrade")
ARCHIVE = os.path.join(REPO, "outputs_limitation_analysis_0faafb9")
DATASETS = ["dtd", "oxford_flowers", "oxford_pets"]


def parse_result_file(path):
    """Yield (label, dataset, acc) from a result txt file."""
    if not os.path.exists(path):
        return
    pat = re.compile(r"^(?P<label>.+?)'s performance on (?P<ds>\S+): Top1- (?P<acc>[0-9.]+)\.$")
    with open(path) as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                yield m.group("label"), m.group("ds"), float(m.group("acc"))


def reference_acc():
    """PTA + CLIP 4-seed means from archived records (recomputed)."""
    ref = {}
    for ds in DATASETS:
        pta, clip = [], []
        for seed in range(1, 5):
            rows = []
            p = os.path.join(ARCHIVE, "records", f"PTA-{ds}-s{seed}", "records.jsonl")
            with open(p) as f:
                for line in f:
                    d = json.loads(line)
                    if d.get("__header__"):
                        continue
                    lg = d.get("logits") or {}
                    if not lg.get("clip"):
                        continue
                    rows.append((np.array(lg["clip"], float), int(d["target"])))
            clip_am = np.array([r[0].argmax() for r in rows])
            tgt = np.array([r[1] for r in rows])
            clip.append(100 * (clip_am == tgt).mean())
        ref[ds] = {"clip": float(np.mean(clip))}
    # PTA from archived result.txt
    pta = defaultdict(list)
    pat = re.compile(r"^PTA-(?P<ds>\S+)-s(?P<seed>[0-9])'s performance on \S+: Top1- (?P<acc>[0-9.]+)\.$")
    with open(os.path.join(ARCHIVE, "result.txt")) as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                pta[m.group("ds")].append(float(m.group("acc")))
    for ds in DATASETS:
        ref[ds]["pta"] = float(np.mean(pta[ds]))
    return ref


def convergence_stats(record_dir):
    """Load convergence.json; return dict or None."""
    p = os.path.join(record_dir, "convergence.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        c = json.load(f)
    return {
        "auc_norm": c.get("auc_norm"),
        "acc10": (c.get("acc_at_fractions") or {}).get("0.1"),
        "acc25": (c.get("acc_at_fractions") or {}).get("0.25"),
        "acc50": (c.get("acc_at_fractions") or {}).get("0.5"),
    }


def main():
    ref = reference_acc()
    print("Reference (4-seed mean, recomputed from archive):")
    for ds in DATASETS:
        print(f"  {ds:16} CLIP={ref[ds]['clip']:.2f}  PTA={ref[ds]['pta']:.2f}")

    for result_file in sorted(glob.glob(os.path.join(OUT, "result_*.txt"))):
        name = os.path.basename(result_file)
        print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
        # group: (setting_suffix, ds) -> [accs]; labels[(setting, ds, seed)] -> exact
        # record-dir label (for convergence lookup, scoped to the right records_* dir)
        groups = defaultdict(list)
        labels = {}
        for label, ds, acc in parse_result_file(result_file):
            # label format: <METHOD>-<setting>-<ds>-s<seed>  e.g. CMP-soft-dtd-s1
            # METHOD may contain digits (BNKV2), so [A-Z0-9]+ not [A-Z]+.
            # ds is constrained to known dataset names (longest first) so that
            # dashed settings like "A0.003-T10" are not split at the first
            # dash (a bare \S+ ds would absorb "T10-dtd" and collapse cells).
            ds_alt = "|".join(sorted(DATASETS, key=len, reverse=True))
            m = re.match(
                r"^(?P<m>[A-Z0-9]+)-(?P<setting>.+?)-(?P<ds>" + ds_alt + r")-s(?P<seed>[0-9])$",
                label,
            )
            if not m:
                print(f"  (unparsed label: {label})")
                continue
            seed = int(m.group("seed"))
            groups[(m.group("setting"), ds)].append((seed, acc))
            labels[(m.group("setting"), ds, seed)] = label
        if not groups:
            print("  (no results)")
            continue
        settings = sorted(set(s for s, _ in groups))
        print(f"  {'setting':14} " + "  ".join(f"{ds[:12]:>14}" for ds in DATASETS) + f"  {'avg(3)':>8}  {'vs PTA':>8}")
        for s in settings:
            row = []
            avgs = []
            for ds in DATASETS:
                vals = [a for _, a in sorted(groups.get((s, ds), []))]
                if len(vals) < 4:
                    row.append(f"{'n='+str(len(vals)):>14}")
                    continue
                a = float(np.mean(vals))
                avgs.append(a)
                row.append(f"{a:14.3f}")
            avg3 = float(np.mean(avgs)) if len(avgs) == 3 else float("nan")
            pta_avg = float(np.mean([ref[ds]["pta"] for ds in DATASETS if (s, ds) in groups and len(groups[(s, ds)]) == 4]))
            delta = avg3 - pta_avg if avgs else float("nan")
            print(f"  {s:14} " + "  ".join(row) + f"  {avg3:8.3f}  {delta:+8.3f}")
        # per-seed detail for the first setting (spread check)
        # convergence — scoped to THIS result file's records_* dir and the exact
        # run label, so e.g. combos "K3" never picks up bank-v1's convergence
        records_sub = "records_" + name[len("result_"):-len(".txt")]
        conv = defaultdict(list)
        for (s, ds), vals in groups.items():
            for seed, _ in vals:
                label = labels.get((s, ds, seed))
                if not label:
                    continue
                c = convergence_stats(os.path.join(OUT, records_sub, label))
                if c:
                    conv[(s, ds)].append(c)
        if conv:
            print(f"\n  Convergence (mean over runs per setting x dataset):")
            print(f"  {'setting':14} {'ds':16} {'n':>3} {'auc_norm':>9} {'acc@10%':>8} {'acc@25%':>8} {'acc@50%':>8}")
            for (s, ds) in sorted(conv):
                cs = conv[(s, ds)]
                n = len(cs)
                auc = float(np.mean([c["auc_norm"] for c in cs if c["auc_norm"] is not None]))
                # acc_at_fractions values are already percentages (e.g. 42.012)
                a10 = float(np.mean([c["acc10"] for c in cs if c["acc10"] is not None]))
                a25 = float(np.mean([c["acc25"] for c in cs if c["acc25"] is not None]))
                a50 = float(np.mean([c["acc50"] for c in cs if c["acc50"] is not None]))
                print(f"  {s:14} {ds:16} {n:3d} {auc:9.4f} {a10:8.2f} {a25:8.2f} {a50:8.2f}")


if __name__ == "__main__":
    main()
