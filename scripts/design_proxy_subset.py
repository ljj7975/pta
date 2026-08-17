#!/usr/bin/env python3
"""
design_proxy_subset.py — seeded, reproducible design of a 10-12 class proxy subset.

Design-time proxy for the subset evaluation: instead of running the combo method
on the full dataset, pick 10-12 classes from the EXISTING full-dataset per-class
records such that

  - the subset spans 3-4 difficulty strata (mean PTA-CS per-class accuracy),
  - no selected class is at the accuracy floor (mean PTA-CS acc < 10%),
  - for DTD: the original 5 difficult classes (bumpy, flecked, lacelike, lined,
    pitted) are never re-selected (they are the old bottom-5 hard subset),
  - the subset's mean combo-PTA delta (the design-time proxy for the combo
    behavior) stays within 1.5 pp of the full-dataset mean delta.

Two dataset modes share the IDENTICAL methodology (same exclusion rule, same
stratification, same seeded tie-break):

  --dataset dtd      (default) DTD proxy subset (Task 2).
        PTA labels:   PTA-CS-s1 / PTA-CS-s2
        combo labels: PatchModPTA-CS-s1 / PatchModPTA-CS-s2
        records:      outputs/records
        out:          outputs/subset_classes_proxy.txt

  --dataset flowers            Oxford-102 proxy subset (Task 7).
        PTA labels:   PTA-CS-flowers-s1 / PTA-CS-flowers-s2
        combo labels: Combo1-flowers-s1 / Combo1-flowers-s2
        records:      outputs/records_proxy_validation
        out:          outputs/flower_subset_classes.txt

Inputs (fully offline, no GPU):
  outputs/records*/<LABEL>/records.jsonl  per-sample records; each sample line
  has ``target`` (int label) + ``correct`` (bool).  The header carries
  ``classnames: []`` (not populated) — the target int is mapped to a classname
  via the dataset class list (see "classname source" below).

Algorithm (single method — no strategy search):
  1. Per class: mean PTA-CS acc across seeds 1+2 (difficulty) and mean
     combo-PTA delta across seeds 1+2.
  2. Exclusion: drop classes with mean PTA-CS acc < 10% (floor).  For DTD also
     drop the original 5 (bumpy, flecked, lacelike, lined, pitted); flowers has
     no legacy hard-subset exclusion.
  3. Stratify: sort the remaining classes by difficulty into 4 equal-size
     strata (3 strata when fewer than 16 candidates remain).  From each
     stratum pick ``count`` classes (3-4 each, totalling --k) whose
     combo-PTA delta is closest to that stratum's median delta
     (tie-break: seeded RNG, then classname).
  4. Write the class file (one classname per line, dataset order, comment-free).
     The subset loader (utils/data.py build_subset_test_data_loader ->
     _parse_class_selection) skips ``#`` comment lines, but the file is kept
     comment-free regardless; the loader matches each line EXACTLY against
     ``dataset.classnames``.
  5. Print per-stratum composition, subset vs full mean delta, and the design
     check |subset - full| <= 1.5 pp.

Classname source (documented on stdout):
  dtd:     1. datasets.build_dataset("dtd").classnames — preferred when
               importable offline (mirrors analyze_records.resolve_classnames).
           2. else: class order parsed from outputs/perclass_tables.md (47 rows).
           3. else: the hardcoded 47-name alphabetical DTD list.
  flowers: 1. datasets.build_dataset("oxford_flowers").classnames — preferred.
           2. else: class order parsed from ./data/oxford_flowers/cat_to_name.json
              (keys sorted numerically — matches dataset label order).
           3. else: the hardcoded 102-name Oxford-102 list (cat_to_name order).

Determinism: ``random.Random(seed)`` for tie-breaking; same args -> identical
output file.  Missing/malformed record lines are skipped and counted.
"""

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import TypedDict


class _DatasetConfig(TypedDict):
    """Typed shape of one ``DATASETS`` entry (per-mode configuration)."""
    key: str
    dataset: str
    label: str
    records: str
    out: str
    pta_labels: list[str]
    patch_labels: list[str]
    expected_records: int
    original_excluded: set[str]

# ---------------------------------------------------------------------------
# Pre-registered constants
# ---------------------------------------------------------------------------

FLOOR_ACC = 10.0  # % mean PTA-CS acc; classes below this are excluded
MAX_STRATUM_PICK = 4  # per-stratum pick cap (spec: 3-4 classes per stratum)

DTD_ORIGINAL_5 = {"bumpy", "flecked", "lacelike", "lined", "pitted"}

PTA_LABELS_DTD = ["PTA-CS-s1", "PTA-CS-s2"]
PATCH_LABELS_DTD = ["PatchModPTA-CS-s1", "PatchModPTA-CS-s2"]

PTA_LABELS_FLOWERS = ["PTA-CS-flowers-s1", "PTA-CS-flowers-s2"]
PATCH_LABELS_FLOWERS = ["Combo1-flowers-s1", "Combo1-flowers-s2"]

# Per-dataset configuration.  The two modes share FLOOR_ACC / stratification /
# seeded tie-break verbatim; only labels, records dir, output path and the
# legacy-exclusion set differ (flowers has no original-5 set).
DATASETS: dict[str, _DatasetConfig] = {
    "dtd": {
        "key": "dtd",
        "dataset": "dtd",
        "label": "DTD",
        "records": "outputs/records",
        "out": "outputs/subset_classes_proxy.txt",
        "pta_labels": PTA_LABELS_DTD,
        "patch_labels": PATCH_LABELS_DTD,
        "expected_records": 1692,  # 47 classes x 36 test images (informational)
        "original_excluded": DTD_ORIGINAL_5,
    },
    "flowers": {
        "key": "flowers",
        "dataset": "oxford_flowers",
        "label": "Oxford Flowers",
        "records": "outputs/records_proxy_validation",
        "out": "outputs/flower_subset_classes.txt",
        "pta_labels": PTA_LABELS_FLOWERS,
        "patch_labels": PATCH_LABELS_FLOWERS,
        "expected_records": 2463,  # 102 classes x ~24-25 test images (informational)
        "original_excluded": set(),
    },
}

# Hardcoded fallback: the 47 DTD classnames in label (alphabetical) order.
DTD_CLASSNAMES_ALPHA = [
    "banded", "blotchy", "braided", "bubbly", "bumpy", "chequered",
    "cobwebbed", "cracked", "crosshatched", "crystalline", "dotted",
    "fibrous", "flecked", "freckled", "frilly", "gauzy", "grid", "grooved",
    "honeycombed", "interlaced", "knitted", "lacelike", "lined", "marbled",
    "matted", "meshed", "paisley", "perforated", "pitted", "pleated",
    "polka-dotted", "porous", "potholed", "scaly", "smeared", "spiralled",
    "sprinkled", "stained", "stratified", "striped", "studded", "swirly",
    "veined", "waffled", "woven", "wrinkled", "zigzagged",
]

# Hardcoded fallback: the 102 Oxford-102 classnames in label order (matches
# cat_to_name.json keys sorted numerically == dataset.classnames order).
FLOWERS_CLASSNAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
    "sweet pea", "english marigold", "tiger lily",
    "moon orchid", "bird of paradise", "monkshood",
    "globe thistle", "snapdragon", "colt's foot",
    "king protea", "spear thistle", "yellow iris",
    "globe-flower", "purple coneflower", "peruvian lily",
    "balloon flower", "giant white arum lily", "fire lily",
    "pincushion flower", "fritillary", "red ginger",
    "grape hyacinth", "corn poppy", "prince of wales feathers",
    "stemless gentian", "artichoke", "sweet william",
    "carnation", "garden phlox", "love in the mist",
    "mexican aster", "alpine sea holly", "ruby-lipped cattleya",
    "cape flower", "great masterwort", "siam tulip",
    "lenten rose", "barbeton daisy", "daffodil",
    "sword lily", "poinsettia", "bolero deep blue",
    "wallflower", "marigold", "buttercup",
    "oxeye daisy", "common dandelion", "petunia",
    "wild pansy", "primula", "sunflower",
    "pelargonium", "bishop of llandaff", "gaura",
    "geranium", "orange dahlia", "pink-yellow dahlia",
    "cautleya spicata", "japanese anemone", "black-eyed susan",
    "silverbush", "californian poppy", "osteospermum",
    "spring crocus", "bearded iris", "windflower",
    "tree poppy", "gazania", "azalea",
    "water lily", "rose", "thorn apple",
    "morning glory", "passion flower", "lotus",
    "toad lily", "anthurium", "frangipani",
    "clematis", "hibiscus", "columbine",
    "desert-rose", "tree mallow", "magnolia",
    "cyclamen", "watercress", "canna lily",
    "hippeastrum", "bee balm", "ball moss",
    "foxglove", "bougainvillea", "camellia",
    "mallow", "mexican petunia", "bromelia",
    "blanket flower", "trumpet creeper", "blackberry lily",
]


# ---------------------------------------------------------------------------
# Record loading (mirrors analyze_records.py, defensively)
# ---------------------------------------------------------------------------

def load_records(records_dir, label):
    """``(header, samples, skipped)`` for ``DIR/label/records.jsonl``.

    Lines that fail to parse as JSON are skipped and counted; ``skipped``
    includes both JSONDecodeError lines and record lines without a usable
    ``target``.
    """
    path = Path(records_dir) / label / "records.jsonl"
    header = None
    samples = []
    skipped = 0
    if not path.is_file():
        return header, samples, skipped
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if rec.get("__header__"):
                header = rec
                continue
            try:
                target = int(rec.get("target", -1))
            except (TypeError, ValueError):
                skipped += 1
                continue
            if target < 0:
                skipped += 1
                continue
            samples.append(rec)
    return header, samples, skipped


def per_class_stats(samples):
    """``{target_idx: {"total", "correct"}}`` recomputed from JSONL."""
    stats = {}
    for s in samples:
        t = int(s.get("target", -1))
        st = stats.setdefault(t, {"total": 0, "correct": 0})
        st["total"] += 1
        if s.get("correct"):
            st["correct"] += 1
    return stats


def class_acc(stats, target):
    st = stats.get(target) or {"total": 0, "correct": 0}
    return 100.0 * st["correct"] / st["total"] if st["total"] else 0.0


# ---------------------------------------------------------------------------
# Classname resolution (documented source, in priority order)
# ---------------------------------------------------------------------------

def classnames_from_datasets(dataset_name):
    """Repo-local ``datasets.build_dataset(<name>).classnames`` (offline).

    The repo vendors its own ``datasets/`` package (and ``clip/``); prepend
    the repo root to ``sys.path`` so the local copy is used regardless of how
    the script is invoked — otherwise ``import datasets`` may resolve to a
    site-packages package with a different ``build_dataset`` signature.
    """
    try:
        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from datasets import build_dataset
        d = build_dataset(dataset_name, "./data")
        cn = [str(c) for c in list(getattr(d, "classnames", []))]
        if cn:
            return cn
    except Exception:
        pass
    return None


def classnames_from_table(table_path):
    """Class order parsed from outputs/perclass_tables.md (47 DTD rows)."""
    if not Path(table_path).is_file():
        return None
    seen = []
    in_table = False
    for line in Path(table_path).read_text(encoding="utf-8").splitlines():
        if line.startswith("| method | seed | class |"):
            in_table = True
            continue
        if not in_table:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 6 or cells[0] not in ("PTA", "PatchModPTA", "ZeroShot"):
            continue
        name = cells[2]
        if name not in seen:
            seen.append(name)
        if len(seen) >= 47:
            break
    return seen or None


def classnames_from_cat_to_name(data_root="./data"):
    """Oxford-102 classnames from ``cat_to_name.json`` (pure stdlib).

    ``cat_to_name.json`` keys are the 1..102 label ids; sorting them
    numerically yields the dataset label order (== dataset.classnames).
    """
    path = Path(data_root) / "oxford_flowers" / "cat_to_name.json"
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        pairs = sorted(raw.items(), key=lambda kv: int(kv[0]))
        cn = [str(c) for _, c in pairs]
    except (ValueError, TypeError, AttributeError):
        return None
    return cn or None


def resolve_classnames(cfg, records_dir, table_path):
    """``(classnames, source_description)`` — first source that succeeds."""
    cn = classnames_from_datasets(cfg["dataset"])
    if cn:
        return cn, "datasets.build_dataset({!r}).classnames (offline import)".format(
            cfg["dataset"])
    if cfg["key"] == "dtd":
        cn = classnames_from_table(table_path)
        if cn:
            return cn, "parsed class order from {} ({} names)".format(
                table_path, len(cn))
        return list(DTD_CLASSNAMES_ALPHA), \
            "hardcoded 47-name alphabetical DTD list"
    cn = classnames_from_cat_to_name()
    if cn:
        return cn, "parsed class order from data/oxford_flowers/cat_to_name.json " + \
                   "({} names)".format(len(cn))
    return list(FLOWERS_CLASSNAMES), \
        "hardcoded 102-name Oxford-102 list (cat_to_name order)"


# ---------------------------------------------------------------------------
# Design
# ---------------------------------------------------------------------------

def compute_class_metrics(records_dir, n_classes, pta_labels, patch_labels):
    """``(pta_mean, delta_mean)`` per target index across all labels.

    ``delta`` = combo acc - PTA-CS acc per seed, averaged over seeds.
    """
    all_labels = pta_labels + patch_labels
    by_label = {}
    for label in all_labels:
        _, samples, _ = load_records(records_dir, label)
        by_label[label] = per_class_stats(samples)

    pta_mean = {}
    delta_mean = {}
    total_by_target = defaultdict(int)
    for t in range(n_classes):
        pta_accs, deltas = [], []
        for seed_label, patch_label in zip(pta_labels, patch_labels):
            p = class_acc(by_label[patch_label], t)
            q = class_acc(by_label[seed_label], t)
            pta_accs.append(q)
            deltas.append(p - q)
            total_by_target[t] += (by_label[patch_label].get(t) or
                                   {"total": 0})["total"]
            total_by_target[t] += (by_label[seed_label].get(t) or
                                   {"total": 0})["total"]
        pta_mean[t] = sum(pta_accs) / len(pta_accs) if pta_accs else 0.0
        delta_mean[t] = sum(deltas) / len(deltas) if deltas else 0.0
    return pta_mean, delta_mean, total_by_target


def median_delta(delta_mean, classes):
    return statistics.median([delta_mean[c] for c in classes])


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Design a stratified 10-12 class proxy subset from existing " +
                    "full-dataset per-class records (offline).")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="dtd",
                        help="dataset mode (dtd or flowers); shares the same " +
                             "exclusion/stratification methodology")
    parser.add_argument("--k", type=int, default=12,
                        help="target subset size (default 12)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for tie-breaking (default 42)")
    parser.add_argument("--records", default=None,
                        help="record dir DIR/<LABEL>/records.jsonl " +
                             "(default per dataset mode)")
    parser.add_argument("--out", default=None,
                        help="output class file (default per dataset mode)")
    args = parser.parse_args(argv)

    cfg = DATASETS[args.dataset]
    k = max(int(args.k), 1)
    seed = int(args.seed)
    rng = random.Random(seed)

    records_dir = Path(args.records) if args.records else Path(cfg["records"])
    out_path = Path(args.out) if args.out else Path(cfg["out"])
    all_labels = cfg["pta_labels"] + cfg["patch_labels"]

    # Classname source: dataset module preferred, per-dataset fallbacks.
    table_candidates = [Path("outputs/perclass_tables.md"),
                        records_dir.parent / "perclass_tables.md"]
    table_path = next((p for p in table_candidates if p.is_file()),
                      table_candidates[0])
    classnames, cn_source = resolve_classnames(cfg, records_dir, table_path)
    n_classes = len(classnames)
    print("[classnames] source: {}".format(cn_source))
    print("[classnames] n = {}".format(n_classes))

    # --- 1. per-class metrics from records.jsonl (target/correct) -----------
    print("[records] loading labels: {}".format(" ".join(all_labels)))
    for label in all_labels:
        _, samples, skipped = load_records(records_dir, label)
        flag = ""
        if len(samples) != cfg["expected_records"]:
            flag = "  <-- expected {} records".format(cfg["expected_records"])
        print("[records] {:24s} n={} skipped={}{}".format(
            label, len(samples), skipped, flag))

    pta_mean, delta_mean, total_by_target = compute_class_metrics(
        records_dir, n_classes, cfg["pta_labels"], cfg["patch_labels"])
    missing = [i for i in range(n_classes) if total_by_target[i] == 0]
    if missing:
        print("[warn] targets with no records: {}".format(missing))

    names = {}  # target index -> classname
    for t in range(n_classes):
        name = classnames[t] if t < len(classnames) else "class_{}".format(t)
        names[t] = name

    # --- 2. exclusion --------------------------------------------------------
    orig_excluded = cfg["original_excluded"]
    floor_excluded = sorted({names[t] for t in range(n_classes)
                             if pta_mean[t] < FLOOR_ACC})
    pool = [t for t in range(n_classes) if pta_mean[t] >= FLOOR_ACC
            and names[t] not in orig_excluded]
    if orig_excluded:
        orig5_excluded = sorted(orig_excluded - {names[t] for t in pool})
        print("\n[exclusion] floor classes (mean PTA-CS acc < {:.0f}%): {}".format(
            FLOOR_ACC, ", ".join(floor_excluded) or "(none)"))
        print("[exclusion] original-5 classes excluded (if not already): {}".format(
            ", ".join(orig5_excluded) or "(none already floor-excluded)"))
    else:
        print("\n[exclusion] floor classes (mean PTA-CS acc < {:.0f}%): {}".format(
            FLOOR_ACC, ", ".join(floor_excluded) or "(none)"))
        print("[exclusion] legacy original-5 exclusion: n/a for {} mode".format(
            args.dataset))
    print("[exclusion] eligible pool: {} classes".format(len(pool)))
    if not pool:
        print(("[ERROR] no eligible classes (all floor-excluded or original-5); " +
               "nothing to design."), file=sys.stderr)
        return 1

    # --- 3. stratify + pick --------------------------------------------------
    pool_sorted = sorted(pool, key=lambda t: (pta_mean[t], names[t]))
    n_strata = 4 if (len(pool_sorted) >= 16 and k >= 8) else 3
    n_strata = min(n_strata, len(pool_sorted), k)
    base_sz, rem_sz = divmod(len(pool_sorted), n_strata)
    strata_sizes = [base_sz + 1] * rem_sz + [base_sz] * (n_strata - rem_sz)
    base_k, rem_k = divmod(k, n_strata)
    pick_counts = [min(base_k + 1, MAX_STRATUM_PICK)] * rem_k + \
                  [min(base_k, MAX_STRATUM_PICK)] * (n_strata - rem_k)

    print("\n[design] strata = {} (difficulty buckets of {} classes)".format(
        n_strata, ",".join(str(s) for s in strata_sizes)))
    print("[design] picks per stratum = {}".format(
        ",".join(str(c) for c in pick_counts)))

    name_width = max((len(names[t]) for t in pool_sorted), default=14) + 1

    selected = []
    cursor = 0
    print("")
    for i in range(n_strata):
        stratum = pool_sorted[cursor:cursor + strata_sizes[i]]
        cursor += strata_sizes[i]
        count = pick_counts[i]
        med = median_delta(delta_mean, stratum)
        lo = pta_mean[stratum[0]]
        hi = pta_mean[stratum[-1]]
        # Closest to the stratum median delta; ties -> seeded RNG, then name.
        keyed = sorted(stratum,
                       key=lambda t: (abs(delta_mean[t] - med),
                                      rng.random(), names[t]))
        picked = keyed[:count]
        selected.extend(picked)
        print(("Stratum {}/{} (difficulty {:.2f}-{:.2f}% PTA acc, " +
               "median delta {:+.2f}pp, pick {}):").format(
                   i + 1, n_strata, lo, hi, med, count))
        for t in stratum:
            mark = "  <--" if t in picked else ""
            print("  {:<{w}s} PTA {:.2f}%  delta {:+.2f}pp{}".format(
                names[t], pta_mean[t], delta_mean[t], mark, w=name_width))

    if len(selected) < k:
        print(("\n[warn] only {} classes available for selection (requested " +
               "{}); proceeding with {}").format(
                   len(selected), k, len(selected)))

    # --- 4. write class file (dataset order) ---------------------------------
    order = {names[t]: t for t in selected}
    selected_names = sorted((names[t] for t in selected),
                            key=lambda n: order[n])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(selected_names) + "\n", encoding="utf-8")

    # --- 5. design-phase validation stats ------------------------------------
    subset_delta = statistics.mean(delta_mean[t] for t in selected)
    full_delta = statistics.mean(delta_mean[t] for t in range(n_classes))
    pool_delta = statistics.mean(delta_mean[t] for t in pool) if pool else 0.0
    diff = abs(subset_delta - full_delta)
    ok = diff <= 1.5

    print("\n== design-phase validation ==")
    print("selected classes ({}): {}".format(
        len(selected_names), ", ".join(selected_names)))
    print("subset mean delta ({:d} classes): {:+.2f} pp".format(
        len(selected_names), subset_delta))
    print("full mean delta ({:d} classes): {:+.2f} pp".format(
        n_classes, full_delta))
    print("eligible-pool mean delta ({:d} classes): {:+.2f} pp".format(
        len(pool), pool_delta))
    print(("design check |subset - full| = {:.2f} pp (PASS if <= 1.50): " +
           "{}").format(diff, "PASS" if ok else "FAIL"))

    # Defensive validation of the output file.
    valid = all(n in classnames for n in selected_names)
    no_orig5 = not (orig_excluded & set(selected_names))
    no_floor = all(pta_mean[order[n]] >= FLOOR_ACC for n in selected_names)
    print(("output validation: all classnames valid={}  " +
           "no original-5={}  no floor-class={}").format(
               valid, no_orig5, no_floor))
    print("[OK] subset classes written to {}".format(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
