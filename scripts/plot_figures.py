#!/usr/bin/env python3
"""plot_figures.py — Unified plotting pipeline for all paper figures.

Consolidates all figure generation from Tasks 2-7 into one script with
subcommands for each figure type.  Reuses analysis logic from the
``analyze_*.py`` modules and generates publication-quality matplotlib/seaborn
plots.

Usage::

    python scripts/plot_figures.py --records DIR --out DIR --figures all
    python scripts/plot_figures.py --records DIR --out DIR --figures perclass,ties
    python scripts/plot_figures.py --records DIR --out DIR --figures all --dry-run

Output: ``<DIR>/fig_<type>.pdf`` and ``<DIR>/fig_<type>.png`` for each figure.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ensure scripts/ is importable for analysis modules
# ---------------------------------------------------------------------------

_SCRIPT_DIR: str = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Matplotlib / seaborn (expected in pta env)
# ---------------------------------------------------------------------------

matplotlib_available: bool = False
plt: Any = None
np: Any = None
sns: Any = None

try:
    import matplotlib as _mpl
    _mpl.use("Agg")
    import matplotlib.pyplot as _plt
    import numpy as _np
    import seaborn as _sns

    plt = _plt
    np = _np
    sns = _sns
    matplotlib_available = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Analysis module imports (self-contained scripts from Tasks 2-7)
# ---------------------------------------------------------------------------

analyze_perclass: Any = None
analyze_ties: Any = None
analyze_spatial: Any = None
analyze_voting_ablation: Any = None
analyze_component_ablation: Any = None
analyze_hyperparam_sensitivity: Any = None

try:
    import analyze_perclass as _mod_ap

    analyze_perclass = _mod_ap
except ImportError:
    pass

try:
    import analyze_ties as _mod_ties

    analyze_ties = _mod_ties
except ImportError:
    pass

try:
    import analyze_spatial as _mod_spatial

    analyze_spatial = _mod_spatial
except ImportError:
    pass

try:
    import analyze_voting_ablation as _mod_vote

    analyze_voting_ablation = _mod_vote
except ImportError:
    pass

try:
    import analyze_component_ablation as _mod_comp

    analyze_component_ablation = _mod_comp
except ImportError:
    pass

try:
    import analyze_hyperparam_sensitivity as _mod_hp

    analyze_hyperparam_sensitivity = _mod_hp
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIGURE_TYPES: List[str] = [
    "perclass",
    "ties",
    "spatial",
    "voting",
    "component",
    "hyperparam",
]

# Consistent color palette for paper figures
COLORS: Dict[str, str] = {
    "pta": "#4C72B0",
    "patch": "#DD8452",
    "positive": "#55A868",
    "negative": "#C44E52",
    "neutral": "#8C8C8C",
    "fusion_qg": "#4C72B0",
    "fusion_mv": "#DD8452",
    "fusion_ag": "#55A868",
    "comp_pta": "#4C72B0",
    "comp_qg": "#DD8452",
    "comp_vote": "#55A868",
}

FONT_SIZES: Dict[str, int] = {
    "title": 14,
    "axis_label": 12,
    "tick": 10,
    "legend": 10,
    "annotation": 9,
}

FIGURE_SIZES: Dict[str, Tuple[float, float]] = {
    "perclass": (8.0, 5.0),
    "ties": (10.0, 5.0),
    "spatial": (7.0, 6.0),
    "voting": (10.0, 5.0),
    "component": (10.0, 5.0),
    "hyperparam": (8.0, 5.0),
}

ABLATION_MARKERS: Tuple[str, ...] = ("-alpha10", "-multigate")


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------


def _apply_style() -> None:
    """Apply consistent matplotlib/seaborn style for publication figures."""
    if not matplotlib_available or plt is None or sns is None:
        return
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "font.size": FONT_SIZES["tick"],
            "axes.titlesize": FONT_SIZES["title"],
            "axes.labelsize": FONT_SIZES["axis_label"],
            "xtick.labelsize": FONT_SIZES["tick"],
            "ytick.labelsize": FONT_SIZES["tick"],
            "legend.fontsize": FONT_SIZES["legend"],
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def _save_figure(fig: Any, out_dir: Path, stem: str) -> None:
    """Save figure as both PDF and PNG, printing Saved: lines."""
    pdf_path = out_dir / "{}.pdf".format(stem)
    png_path = out_dir / "{}.png".format(stem)
    fig.savefig(str(pdf_path))
    fig.savefig(str(png_path))
    plt.close(fig)  # noqa: F541
    print("Saved: {}".format(pdf_path))
    print("Saved: {}".format(png_path))


def _argmax(vec: Any) -> Optional[int]:
    """Index of max element; ``None`` for empty lists."""
    if not vec:
        return None
    return max(range(len(vec)), key=lambda i: vec[i])


# ---------------------------------------------------------------------------
# Record loading helpers (lightweight, used when analysis module unavailable)
# ---------------------------------------------------------------------------


def _list_labels(records_dir: Path) -> List[str]:
    """Sorted subdirs of *records_dir* containing ``records.jsonl``."""
    if not records_dir.is_dir():
        return []
    return sorted(
        sub.name
        for sub in records_dir.iterdir()
        if sub.is_dir() and (sub / "records.jsonl").is_file()
    )


def _load_records(records_dir: Path, label: str) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return ``(header, samples)`` for ``DIR/label/records.jsonl``."""
    path = records_dir / label / "records.jsonl"
    header: Optional[Dict[str, Any]] = None
    samples: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec: Dict[str, Any] = json.loads(line)
                if rec.get("__header__"):
                    header = rec
                else:
                    samples.append(rec)
    except FileNotFoundError:
        print("[WARN] records file not found: {}".format(path), file=sys.stderr)
    except OSError as e:
        print("[WARN] error reading records file {}: {}".format(path, e),
              file=sys.stderr)
    return header, samples


def _overall_acc(samples: List[Dict[str, Any]]) -> Optional[float]:
    """Overall accuracy as percentage (None if empty)."""
    if not samples:
        return None
    return 100.0 * sum(1 for s in samples if s.get("correct")) / len(samples)


# ---------------------------------------------------------------------------
# Figure 1: perclass — horizontal bar chart of accuracy delta per class
# ---------------------------------------------------------------------------


def _extract_perclass_data(
    records_dir: Path,
) -> Optional[List[Dict[str, Any]]]:
    """Extract per-class PTA vs PatchModPTA accuracy and delta.

    Returns list of ``{"class", "pta_acc", "patch_acc", "delta"}`` dicts,
    or ``None`` when data is unavailable.
    """
    if analyze_perclass is None:
        print("[INFO] analyze_perclass not importable — skipping perclass figure")
        return None

    labels: List[str] = analyze_perclass.list_labels(str(records_dir))
    if not labels:
        return None

    headers: List[Any] = [
        analyze_perclass.load_label_records(str(records_dir), label)[0]
        for label in labels
    ]
    classnames: List[str] = analyze_perclass.resolve_classnames(
        str(records_dir), labels, headers
    )

    pta_runs: List[Any] = analyze_perclass.collect_method_samples(
        str(records_dir), labels, analyze_perclass.METHOD_PTA
    )
    patch_runs: List[Any] = analyze_perclass.collect_method_samples(
        str(records_dir), labels, analyze_perclass.METHOD_PATCH
    )

    if not pta_runs or not patch_runs:
        return None

    # Separate main PatchModPTA from voting variants
    patch_main: List[Any] = [
        (l, h, s)
        for l, h, s in patch_runs
        if not analyze_perclass.is_voting_variant(l)
    ]
    if not patch_main:
        patch_main = patch_runs

    pta_by_label: Dict[str, Any] = {}
    for label, _header, samples in pta_runs:
        pta_by_label[label] = analyze_perclass.per_class_named(samples, classnames)

    patch_by_label: Dict[str, Any] = {}
    for label, _header, samples in patch_main:
        patch_by_label[label] = analyze_perclass.per_class_named(samples, classnames)

    pta_means: Dict[str, float] = analyze_perclass._class_means(pta_by_label)
    patch_means: Dict[str, float] = analyze_perclass._class_means(patch_by_label)

    all_classes: List[str] = sorted(
        set(pta_means.keys()) | set(patch_means.keys()),
        key=analyze_perclass._name_sort_key,
    )

    rows: List[Dict[str, Any]] = []
    for name in all_classes:
        p_acc = pta_means.get(name)
        pa_acc = patch_means.get(name)
        delta = (pa_acc - p_acc) if (p_acc is not None and pa_acc is not None) else None
        rows.append(
            {"class": name, "pta_acc": p_acc, "patch_acc": pa_acc, "delta": delta}
        )

    return rows


def figure_perclass(
    records_dir: Path, out_dir: Path, dry_run: bool
) -> Optional[Path]:
    """Bar chart of accuracy delta per class, sorted, hard classes highlighted."""
    if not matplotlib_available or plt is None or np is None:
        print("[INFO] matplotlib not available — skipping perclass figure")
        return None

    rows = _extract_perclass_data(records_dir)
    if not rows:
        print("[INFO] No perclass data found — skipping")
        return None

    if dry_run:
        print("[DRY-RUN] perclass data validated: {} classes".format(len(rows)))
        return None

    _apply_style()

    valid_rows = [r for r in rows if r["delta"] is not None]
    if not valid_rows:
        print("[INFO] No valid deltas for perclass — skipping")
        return None

    valid_rows.sort(key=lambda r: float(r["delta"]))  # noqa: F401

    classes: List[str] = [r["class"] for r in valid_rows]
    deltas: List[float] = [r["delta"] for r in valid_rows]
    colors: List[str] = [
        COLORS["negative"] if d < 0 else COLORS["positive"] if d > 0 else COLORS["neutral"]
        for d in deltas
    ]

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["perclass"])
    y_pos = np.arange(len(classes))
    ax.barh(y_pos, deltas, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Accuracy Delta (pp)")
    ax.set_title("Per-Class Accuracy Delta: PatchModPTA \u2212 PTA")
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    ax.invert_yaxis()

    fig.tight_layout()
    _save_figure(fig, out_dir, "fig_perclass")
    return out_dir / "fig_perclass.pdf"


# ---------------------------------------------------------------------------
# Figure 2: ties — grouped bar chart PTA vs PatchModPTA on ties/non-ties
# ---------------------------------------------------------------------------


def _extract_ties_data(
    records_dir: Path,
) -> Optional[List[Dict[str, Any]]]:
    """Extract ties vs non-ties accuracy per dataset.

    Returns list of dicts with dataset, n_ties, pta/patch acc on ties/non-ties.
    """
    if analyze_ties is None:
        print("[INFO] analyze_ties not importable — skipping ties figure")
        return None

    labels: List[str] = analyze_ties.list_labels(str(records_dir))
    if not labels:
        return None

    pta_by_ds: Dict[str, Dict[int, Any]] = analyze_ties._samples_by_dataset(
        str(records_dir), labels, analyze_ties.METHOD_PTA
    )
    patch_by_ds: Dict[str, Dict[int, Any]] = analyze_ties._samples_by_dataset(
        str(records_dir), labels, analyze_ties.METHOD_PATCH
    )

    all_datasets: List[str] = sorted(
        set(list(pta_by_ds.keys()) + list(patch_by_ds.keys()))
    )
    if not all_datasets:
        return None

    def _subset_acc(
        idxs: Any, idx_map: Dict[int, Any]
    ) -> Optional[float]:
        """Accuracy on a subset of batch indices."""
        idx_list = list(idxs)
        if not idx_list or not idx_map:
            return None
        correct = sum(
            1 for i in idx_list if idx_map.get(i, {}).get("correct")
        )
        return 100.0 * correct / len(idx_list)

    rows: List[Dict[str, Any]] = []
    for ds in all_datasets:
        pta_idx: Dict[int, Any] = pta_by_ds.get(ds, {})
        patch_idx: Dict[int, Any] = patch_by_ds.get(ds, {})
        ref: Dict[int, Any] = pta_idx or patch_idx
        if not ref:
            continue

        tie_idxs = set(
            idx for idx, s in ref.items() if analyze_ties.is_tie(s)
        )
        non_tie_idxs = set(ref.keys()) - tie_idxs
        total: int = len(ref)

        rows.append(
            {
                "dataset": ds,
                "total": total,
                "n_ties": len(tie_idxs),
                "tie_rate": 100.0 * len(tie_idxs) / total if total else 0.0,
                "pta_tie_acc": _subset_acc(tie_idxs, pta_idx),
                "pta_ntie_acc": _subset_acc(non_tie_idxs, pta_idx),
                "patch_tie_acc": _subset_acc(tie_idxs, patch_idx),
                "patch_ntie_acc": _subset_acc(non_tie_idxs, patch_idx),
            }
        )

    return rows


def figure_ties(
    records_dir: Path, out_dir: Path, dry_run: bool
) -> Optional[Path]:
    """Grouped bar chart: accuracy on ties vs non-ties for PTA and PatchModPTA."""
    if not matplotlib_available or plt is None or np is None:
        print("[INFO] matplotlib not available — skipping ties figure")
        return None

    rows = _extract_ties_data(records_dir)
    if not rows:
        print("[INFO] No ties data found — skipping")
        return None

    if dry_run:
        print("[DRY-RUN] ties data validated: {} dataset(s)".format(len(rows)))
        return None

    _apply_style()

    n_ds = len(rows)
    fig, axes = plt.subplots(
        1,
        max(n_ds, 1),
        figsize=FIGURE_SIZES["ties"],
        squeeze=False,
        sharey=True,
    )

    bar_width = 0.35

    for i, r in enumerate(rows):
        ax = axes[0][i]
        categories = ["Ties", "Non-Ties"]
        pta_vals = [r["pta_tie_acc"] or 0.0, r["pta_ntie_acc"] or 0.0]
        patch_vals = [r["patch_tie_acc"] or 0.0, r["patch_ntie_acc"] or 0.0]

        x = np.arange(len(categories))
        bars1 = ax.bar(
            x - bar_width / 2,
            pta_vals,
            bar_width,
            label="PTA",
            color=COLORS["pta"],
        )
        bars2 = ax.bar(
            x + bar_width / 2,
            patch_vals,
            bar_width,
            label="PatchModPTA",
            color=COLORS["patch"],
        )

        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("{}".format(r["dataset"]))
        ax.set_ylim(0, 110)
        ax.legend()

        # Value annotations
        for bar in bars1:
            h = bar.get_height()
            ax.annotate(
                "{:.1f}".format(h),
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZES["annotation"],
            )
        for bar in bars2:
            h = bar.get_height()
            ax.annotate(
                "{:.1f}".format(h),
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZES["annotation"],
            )

    fig.suptitle(
        "Accuracy on Ties vs Non-Ties", fontsize=FONT_SIZES["title"]
    )
    fig.tight_layout()
    _save_figure(fig, out_dir, "fig_ties")
    return out_dir / "fig_ties.pdf"


# ---------------------------------------------------------------------------
# Figure 3: spatial — confusion matrix image_proto vs patch_proto
# ---------------------------------------------------------------------------


def _extract_spatial_data(
    records_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Build confusion matrix of image_proto vs patch_proto argmax, colored by quality_gate.

    Returns dict with ``counts`` (NxN int list), ``qg_avg`` (NxN float list),
    ``num_classes`` (int), or ``None`` when data is unavailable.
    """
    labels: List[str] = _list_labels(records_dir)
    if not labels:
        return None

    # Collect all PatchModPTA samples
    samples: List[Dict[str, Any]] = []
    for label in labels:
        if not label.startswith("PatchModPTA-CS"):
            continue
        if any(m in label for m in ABLATION_MARKERS):
            continue
        _, label_samples = _load_records(records_dir, label)
        samples.extend(label_samples)

    if not samples:
        return None

    # Determine number of classes from target values
    num_classes: int = 0
    for s in samples:
        t = s.get("target")
        if t is not None:
            num_classes = max(num_classes, int(t) + 1)

    if num_classes == 0:
        return None

    # Build confusion matrix: rows=image_proto argmax, cols=patch_proto argmax
    count_matrix: List[List[int]] = [[0] * num_classes for _ in range(num_classes)]
    qg_sum: List[List[float]] = [
        [0.0] * num_classes for _ in range(num_classes)
    ]
    qg_count: List[List[int]] = [[0] * num_classes for _ in range(num_classes)]

    for s in samples:
        lg: Dict[str, Any] = s.get("logits") or {}
        img_proto = lg.get("image_proto")
        patch_proto = lg.get("patch_proto")
        if not img_proto or not patch_proto:
            continue

        img_pred = _argmax(img_proto)
        patch_pred = _argmax(patch_proto)
        if img_pred is None or patch_pred is None:
            continue
        if img_pred >= num_classes or patch_pred >= num_classes:
            continue

        count_matrix[img_pred][patch_pred] += 1
        qg = s.get("quality_gate")
        if qg is not None:
            qg_sum[img_pred][patch_pred] += float(qg)
            qg_count[img_pred][patch_pred] += 1

    # Average quality_gate per cell
    qg_avg: List[List[float]] = [[0.0] * num_classes for _ in range(num_classes)]
    for i in range(num_classes):
        for j in range(num_classes):
            c = qg_count[i][j]
            qg_avg[i][j] = qg_sum[i][j] / c if c > 0 else 0.0

    return {
        "counts": count_matrix,
        "qg_avg": qg_avg,
        "num_classes": num_classes,
    }


def figure_spatial(
    records_dir: Path, out_dir: Path, dry_run: bool
) -> Optional[Path]:
    """Confusion matrix of image_proto vs patch_proto, colored by quality_gate."""
    if not matplotlib_available or plt is None or sns is None:
        print("[INFO] matplotlib/seaborn not available — skipping spatial figure")
        return None

    data = _extract_spatial_data(records_dir)
    if not data:
        print("[INFO] No spatial data found — skipping")
        return None

    if dry_run:
        print(
            "[DRY-RUN] spatial data validated: {} classes".format(
                data["num_classes"]
            )
        )
        return None

    _apply_style()

    num_classes = data["num_classes"]
    counts = data["counts"]
    qg_avg = data["qg_avg"]

    # Create class labels
    class_labels = ["class_{}".format(i) for i in range(num_classes)]

    # Two-panel figure: count matrix + quality_gate matrix
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.0, 5.0), gridspec_kw={"width_ratios": [1, 1]}
    )

    # Panel 1: Count heatmap
    counts_arr = np.array(counts)
    sns.heatmap(
        counts_arr,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax1,
        cbar_kws={"label": "Count"},
    )
    ax1.set_xlabel("patch_proto predicted")
    ax1.set_ylabel("image_proto predicted")
    ax1.set_title("Sample Count")

    # Panel 2: Quality gate heatmap
    qg_arr = np.array(qg_avg)
    # Mask cells with zero count for cleaner visualization
    mask = np.array(counts) == 0
    sns.heatmap(
        qg_arr,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax2,
        mask=mask,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Mean quality_gate"},
    )
    ax2.set_xlabel("patch_proto predicted")
    ax2.set_ylabel("image_proto predicted")
    ax2.set_title("Mean Quality Gate")

    fig.suptitle(
        "Spatial Sensitivity: image_proto vs patch_proto",
        fontsize=FONT_SIZES["title"],
    )
    fig.tight_layout()
    _save_figure(fig, out_dir, "fig_spatial")
    return out_dir / "fig_spatial.pdf"


# ---------------------------------------------------------------------------
# Figure 4: voting — grouped bar chart accuracy per fusion strategy
# ---------------------------------------------------------------------------


def _extract_voting_data(
    records_dir: Path,
) -> Optional[List[Dict[str, Any]]]:
    """Extract per-dataset accuracy for each fusion strategy.

    Returns list of dicts with dataset, seed, acc per fusion strategy.
    """
    if analyze_voting_ablation is None:
        print(
            "[INFO] analyze_voting_ablation not importable — skipping voting figure"
        )
        return None

    groups: Dict[Any, Any]
    classnames_map: Dict[Any, Any]
    groups, classnames_map = analyze_voting_ablation.analyze_records(
        str(records_dir)
    )
    if not groups:
        return None

    rows = analyze_voting_ablation.build_comparison_table(groups)
    return rows


def figure_voting(
    records_dir: Path, out_dir: Path, dry_run: bool
) -> Optional[Path]:
    """Grouped bar chart: accuracy per fusion strategy per dataset."""
    if not matplotlib_available or plt is None or np is None:
        print("[INFO] matplotlib not available — skipping voting figure")
        return None

    rows = _extract_voting_data(records_dir)
    if not rows:
        print("[INFO] No voting data found — skipping")
        return None

    if dry_run:
        print("[DRY-RUN] voting data validated: {} row(s)".format(len(rows)))
        return None

    _apply_style()

    # Group by dataset (average across seeds)
    ds_data: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        ds = r["dataset"]
        if r.get("acc_QualityGated") is not None:
            ds_data[ds]["QualityGated"].append(r["acc_QualityGated"])
        if r.get("acc_MajorityVote") is not None:
            ds_data[ds]["MajorityVote"].append(r["acc_MajorityVote"])
        if r.get("acc_AgreementGate") is not None:
            ds_data[ds]["AgreementGate"].append(r["acc_AgreementGate"])

    datasets = sorted(ds_data.keys())
    if not datasets:
        print("[INFO] No voting datasets — skipping")
        return None

    strategies = ["QualityGated", "MajorityVote", "AgreementGate"]
    strat_colors = [COLORS["fusion_qg"], COLORS["fusion_mv"], COLORS["fusion_ag"]]
    strat_labels = ["QualityGated", "MajorityVote", "AgreementGate"]

    n_ds = len(datasets)
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["voting"])

    x = np.arange(n_ds)
    bar_width = 0.25

    for si, (strat, color, label) in enumerate(
        zip(strategies, strat_colors, strat_labels)
    ):
        vals: List[float] = []
        for ds in datasets:
            accs = ds_data[ds].get(strat, [])
            vals.append(sum(accs) / len(accs) if accs else 0.0)
        offset = (si - 1) * bar_width
        bars = ax.bar(
            x + offset, vals, bar_width, label=label, color=color
        )
        # Value annotations
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(
                    "{:.1f}".format(h),
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=FONT_SIZES["annotation"],
                )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Fusion Strategy Comparison")
    ax.legend()
    ax.set_ylim(0, 110)

    fig.tight_layout()
    _save_figure(fig, out_dir, "fig_voting")
    return out_dir / "fig_voting.pdf"


# ---------------------------------------------------------------------------
# Figure 5: component — stacked bar chart cumulative contribution
# ---------------------------------------------------------------------------


def _extract_component_data(
    records_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Extract per-dataset accuracy for each component configuration.

    Returns dict with ``dataset_results`` and ``config_names``.
    """
    if analyze_component_ablation is None:
        print(
            "[INFO] analyze_component_ablation not importable "
            "— skipping component figure"
        )
        return None

    dataset_results: Dict[str, Any]
    all_labels: Dict[str, Any]
    dataset_results, all_labels = analyze_component_ablation.analyze(
        str(records_dir)
    )
    if not dataset_results:
        return None

    config_names = [c[0] for c in analyze_component_ablation.CONFIGS]
    return {"dataset_results": dataset_results, "config_names": config_names}


def figure_component(
    records_dir: Path, out_dir: Path, dry_run: bool
) -> Optional[Path]:
    """Stacked bar chart showing cumulative component contribution."""
    if not matplotlib_available or plt is None or np is None:
        print("[INFO] matplotlib not available — skipping component figure")
        return None

    data = _extract_component_data(records_dir)
    if not data:
        print("[INFO] No component data found — skipping")
        return None

    if dry_run:
        n_ds = len(data["dataset_results"])
        print("[DRY-RUN] component data validated: {} dataset(s)".format(n_ds))
        return None

    _apply_style()

    dataset_results = data["dataset_results"]
    config_names = data["config_names"]
    sorted_datasets = sorted(dataset_results.keys())

    # Compute mean accuracy per config per dataset
    # Config order: PTA (baseline), PatchModPTA+QG, PatchModPTA+Vote
    comp_colors = [COLORS["comp_pta"], COLORS["comp_qg"], COLORS["comp_vote"]]
    comp_labels = ["PTA (baseline)", "PatchModPTA+QG", "PatchModPTA+Vote"]

    # For stacked bar: show baseline as bottom, then deltas
    baselines: List[float] = []
    delta_qg: List[float] = []
    delta_vote: List[float] = []

    for ds in sorted_datasets:
        ds_data = dataset_results[ds]
        accs: Dict[str, List[float]] = {}
        for cn in config_names:
            info = ds_data.get(cn)
            if info and info.get("accs"):
                accs[cn] = info["accs"]
            else:
                accs[cn] = []

        pta_acc = (
            sum(accs["PTA"]) / len(accs["PTA"]) if accs["PTA"] else 0.0
        )
        qg_acc = (
            sum(accs["PatchModPTA+QG"]) / len(accs["PatchModPTA+QG"])
            if accs["PatchModPTA+QG"]
            else pta_acc
        )
        vote_acc = (
            sum(accs["PatchModPTA+Vote"]) / len(accs["PatchModPTA+Vote"])
            if accs["PatchModPTA+Vote"]
            else qg_acc
        )

        baselines.append(pta_acc)
        delta_qg.append(max(qg_acc - pta_acc, 0.0))
        delta_vote.append(max(vote_acc - qg_acc, 0.0))

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["component"])

    x = np.arange(len(sorted_datasets))
    bar_width = 0.6

    # Stacked bars
    ax.bar(
        x,
        baselines,
        bar_width,
        label=comp_labels[0],
        color=comp_colors[0],
    )
    ax.bar(
        x,
        delta_qg,
        bar_width,
        bottom=baselines,
        label=comp_labels[1],
        color=comp_colors[1],
    )
    bottoms = [b + d for b, d in zip(baselines, delta_qg)]
    ax.bar(
        x,
        delta_vote,
        bar_width,
        bottom=bottoms,
        label=comp_labels[2],
        color=comp_colors[2],
    )

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_datasets, rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Component Ablation: Cumulative Contribution")
    ax.legend()
    ax.set_ylim(0, 110)

    fig.tight_layout()
    _save_figure(fig, out_dir, "fig_component")
    return out_dir / "fig_component.pdf"


# ---------------------------------------------------------------------------
# Figure 6: hyperparam — line plots accuracy vs hyperparameter value
# ---------------------------------------------------------------------------


def _extract_hyperparam_data(
    records_dir: Path,
) -> Optional[List[Dict[str, Any]]]:
    """Extract sweep results for each hyperparameter.

    Returns list of sweep result dicts with ``param_key``, ``grouped``,
    ``has_variation``.
    """
    if analyze_hyperparam_sensitivity is None:
        print(
            "[INFO] analyze_hyperparam_sensitivity not importable "
            "— skipping hyperparam figure"
        )
        return None

    sweep_results: List[Dict[str, Any]] = []
    for param_key, param_info in analyze_hyperparam_sensitivity.SWEEP_PARAMS.items():
        result = analyze_hyperparam_sensitivity.analyze_sweep(
            str(records_dir), param_key, param_info
        )
        sweep_results.append(result)

    return sweep_results


def figure_hyperparam(
    records_dir: Path, out_dir: Path, dry_run: bool
) -> Optional[Path]:
    """Line plots: accuracy vs hyperparameter value, one line per dataset."""
    if not matplotlib_available or plt is None or np is None:
        print("[INFO] matplotlib not available — skipping hyperparam figure")
        return None

    sweep_results = _extract_hyperparam_data(records_dir)
    if not sweep_results:
        print("[INFO] No hyperparam data found — skipping")
        return None

    plot_params = [r for r in sweep_results if r["has_variation"]]
    if not plot_params:
        if dry_run:
            print(
                "[DRY-RUN] hyperparam data validated: "
                "{} param(s) found, no sweep variation".format(
                    len(sweep_results)
                )
            )
        else:
            print("[INFO] No sweep variation detected — skipping hyperparam plots")
        return None

    if dry_run:
        print(
            "[DRY-RUN] hyperparam data validated: "
            "{} sweep param(s)".format(len(plot_params))
        )
        return None

    _apply_style()

    n = len(plot_params)
    fig, axes_arr = plt.subplots(
        1, max(n, 1), figsize=(6.0 * max(n, 1), 5.0), squeeze=False
    )

    for idx, result in enumerate(plot_params):
        ax = axes_arr[0][idx]
        key = result["param_key"]

        # Collect per-dataset accuracy vs value
        ds_by_value: Dict[Any, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for label_str in _list_labels(records_dir):
            header, samples = _load_records(records_dir, label_str)
            if not header or not samples:
                continue
            rc: Dict[str, Any] = header.get("resolved_config") or {}

            # Resolve dotted key
            value: Any = None
            keys = key.split(".")
            d: Any = rc
            for k in keys:
                if not isinstance(d, dict):
                    d = None
                    break
                d = d.get(k)
            if d is not None:
                value = d
            else:
                leaf = keys[-1]
                value = rc.get(leaf)

            if value is None:
                continue

            # Normalize to float when possible
            try:
                num_val = round(float(value), 10)
            except (TypeError, ValueError):
                num_val = value

            dataset = header.get("dataset", "unknown")
            acc = _overall_acc(samples)
            if acc is not None:
                ds_by_value[num_val][dataset].append(acc)

        # Plot one line per dataset
        all_ds = sorted(
            set(
                ds
                for vds in ds_by_value.values()
                for ds in vds
            )
        )
        for ds in all_ds:
            xs: List[float] = []
            ys: List[float] = []
            for val in sorted(ds_by_value.keys()):
                accs = ds_by_value[val].get(ds, [])
                if accs:
                    try:
                        num = float(val)
                        xs.append(num)
                        ys.append(sum(accs) / len(accs))
                    except (TypeError, ValueError):
                        pass
            if xs:
                ax.plot(xs, ys, "o-", label=ds, linewidth=2, markersize=6)

        short = key.rsplit(".", 1)[-1]
        ax.set_xlabel(short)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy vs {}".format(key))
        ax.legend(fontsize=FONT_SIZES["legend"], loc="best")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, out_dir, "fig_hyperparam")
    return out_dir / "fig_hyperparam.pdf"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

FIGURE_DISPATCH: Dict[str, Any] = {
    "perclass": figure_perclass,
    "ties": figure_ties,
    "spatial": figure_spatial,
    "voting": figure_voting,
    "component": figure_component,
    "hyperparam": figure_hyperparam,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified plotting pipeline for all paper figures."
    )
    parser.add_argument(
        "--records",
        required=True,
        help="Record directory (DIR/<LABEL>/records.jsonl)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for figures (PDF + PNG)",
    )
    parser.add_argument(
        "--figures",
        default="all",
        help=(
            "Comma-separated figure types to generate, or 'all'. "
            "Choices: {} (default: all)".format(",".join(FIGURE_TYPES))
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data exists without generating plots",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    records_dir = Path(args.records)
    if not records_dir.is_dir():
        print(
            "[ERROR] records dir not found: {}".format(records_dir),
            file=sys.stderr,
        )
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse figure types
    if args.figures == "all":
        figure_list = list(FIGURE_TYPES)
    else:
        figure_list = [f.strip() for f in args.figures.split(",")]
        invalid = [f for f in figure_list if f not in FIGURE_TYPES]
        if invalid:
            print(
                "[ERROR] unknown figure type(s): {}. Choices: {}".format(
                    ", ".join(invalid), ", ".join(FIGURE_TYPES)
                ),
                file=sys.stderr,
            )
            return 1

    if not matplotlib_available:
        print(
            "[WARNING] matplotlib not installed — no figures will be generated",
            file=sys.stderr,
        )

    # Generate each figure
    generated: List[str] = []
    skipped: List[str] = []

    for fig_type in figure_list:
        func = FIGURE_DISPATCH.get(fig_type)
        if func is None:
            print("[WARNING] unknown figure type: {}".format(fig_type))
            skipped.append(fig_type)
            continue

        result = func(records_dir, out_dir, args.dry_run)
        if result is not None:
            generated.append(fig_type)
        else:
            skipped.append(fig_type)

    # Summary
    print("")
    print("--- Summary ---")
    if args.dry_run:
        print("Dry-run mode: data validated, no plots generated")
    if generated:
        print("Generated: {}".format(", ".join(generated)))
    if skipped:
        print("Skipped: {}".format(", ".join(skipped)))

    return 0


if __name__ == "__main__":
    sys.exit(main())
