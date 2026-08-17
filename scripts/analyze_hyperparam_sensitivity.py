#!/usr/bin/env python3
"""Hyperparameter sensitivity analysis (Exp 4.2).

Analyzes existing records from hyperparameter sweep runs (if available).
Groups records by key hyperparameter values and computes accuracy per group.

Key hyperparameters analyzed (defaults from ``configs/base.yaml``):
  - ``image_level.alpha`` (0.001, 0.01, 0.1, 1.0)
  - ``image_level.T`` (10, 20, 50, 100)
  - ``patch_level.conf_threshold`` (0.1, 0.3, 0.5, 0.7)

When no sweep variation is detected (all records share the same value),
the script warns and produces a placeholder report suggesting how to
generate sweep data using ``runner.py --override``.

Usage:
    python scripts/analyze_hyperparam_sensitivity.py \\
        --records tests/fixtures/records_sample/ \\
        --out outputs/hyperparam_sensitivity.md

    # Filter to a single hyperparameter:
    python scripts/analyze_hyperparam_sensitivity.py \\
        --records tests/fixtures/records_sample/ \\
        --sweep image_level.alpha \\
        --out outputs/hyperparam_alpha.md
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

matplotlib_available = False
plt: Any = None
try:
    import matplotlib as _mpl
    _mpl.use("Agg")
    import matplotlib.pyplot as _plt  # noqa: F811
    plt = _plt
    matplotlib_available = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Sweep parameter registry
# ---------------------------------------------------------------------------

SWEEP_PARAMS: Dict[str, Dict[str, Any]] = {
    "image_level.alpha": {
        "description": "Weight of original text features vs. updated prototype",
        "default": 0.01,
        "expected_values": [0.001, 0.01, 0.1, 1.0],
    },
    "image_level.T": {
        "description": "Temperature for exponential update weight",
        "default": 20.0,
        "expected_values": [10, 20, 50, 100],
    },
    "patch_level.conf_threshold": {
        "description": "Confidence threshold for patch filtering",
        "default": 0.5,
        "expected_values": [0.1, 0.3, 0.5, 0.7],
    },
}


# ---------------------------------------------------------------------------
# Record loading (follows analyze_records.py pattern)
# ---------------------------------------------------------------------------

def list_labels(records_dir: str) -> List[str]:
    """Sorted labels = subdirs of ``records_dir`` containing ``records.jsonl``."""
    d = Path(records_dir)
    if not d.is_dir():
        return []
    return sorted(
        sub.name for sub in d.iterdir()
        if sub.is_dir() and (sub / "records.jsonl").is_file()
    )


def load_label_records(records_dir: str, label: str) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return ``(header, samples)`` for ``DIR/label/records.jsonl``."""
    path = Path(records_dir) / label / "records.jsonl"
    header: Optional[Dict[str, Any]] = None
    samples: List[Dict[str, Any]] = []
    try:
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
    except FileNotFoundError:
        print("[WARN] records file not found: {}".format(path), file=sys.stderr)
    except OSError as e:
        print("[WARN] error reading records file {}: {}".format(path, e),
              file=sys.stderr)
    return header, samples


def overall_acc(samples: List[Dict[str, Any]]) -> Optional[float]:
    """Overall accuracy as a percentage float (None if empty)."""
    if not samples:
        return None
    return 100.0 * sum(1 for s in samples if s.get("correct")) / len(samples)


# ---------------------------------------------------------------------------
# Config value resolution
# ---------------------------------------------------------------------------

def resolve_config_value(resolved_config: Any, dotted_key: str) -> Any:
    """Resolve a dotted key from a (possibly nested) config dict.

    Tries the exact dotted path first (``image_level.alpha``), then
    falls back to leaf-only lookup (``alpha``) for flattened configs.
    Returns ``None`` when the key is absent.
    """
    if not isinstance(resolved_config, dict):
        return None
    keys = dotted_key.split(".")
    d: Any = resolved_config
    for k in keys:
        if not isinstance(d, dict):
            d = None
            break
        d = d.get(k)
    if d is not None:
        return d
    leaf = keys[-1]
    return resolved_config.get(leaf)


def _to_float(v: Any) -> Optional[float]:
    """Safely cast to float; returns None on failure."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Sweep analysis
# ---------------------------------------------------------------------------

def analyze_sweep(records_dir: str, param_key: str,
                   param_info: Dict[str, Any]) -> Dict[str, Any]:
    """Group records by the value of *param_key* and compute accuracy."""
    labels = list_labels(records_dir)

    grouped: Dict[Any, Dict[str, Any]] = defaultdict(
        lambda: {"labels": [], "accs": []})

    for label in labels:
        header, samples = load_label_records(records_dir, label)
        if not header or not samples:
            continue

        resolved_config = header.get("resolved_config") or {}
        value = resolve_config_value(resolved_config, param_key)

        if value is None:
            grouped["unknown"]["labels"].append(label)
            grouped["unknown"]["accs"].append(overall_acc(samples))
            continue

        num_val = _to_float(value)
        if num_val is not None:
            norm_value: Any = round(num_val, 10)
        else:
            norm_value = str(value)

        acc = overall_acc(samples)
        grouped[norm_value]["labels"].append(label)
        if acc is not None:
            grouped[norm_value]["accs"].append(acc)

    real_values = [v for v in grouped if v != "unknown"]
    has_variation = len(real_values) > 1

    result_grouped: Dict[Any, Dict[str, Any]] = {}
    for value in sorted(grouped.keys(), key=lambda v: (str(v) != "unknown", v)):
        g = grouped[value]
        accs = [a for a in g["accs"] if a is not None]
        result_grouped[value] = {
            "labels": list(g["labels"]),
            "accs": list(accs),
            "mean_acc": sum(accs) / len(accs) if accs else None,
            "n_seeds": len(accs),
        }

    return {
        "param_key": param_key,
        "param_info": param_info,
        "grouped": result_grouped,
        "has_variation": has_variation,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(v: Any, precision: int = 2) -> str:
    if v is None:
        return "\u2014"
    if isinstance(v, float):
        return "{:.4f}".format(v) if abs(v) < 0.1 else "{:.{p}f}".format(v, p=precision)
    return str(v)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(records_dir: str, sweep_results: List[Dict[str, Any]]) -> List[str]:
    """Build markdown report lines."""
    lines: List[str] = []
    lines.append("# Hyperparameter Sensitivity (Exp 4.2)")
    lines.append("")
    lines.append("Analyzes accuracy across hyperparameter sweep values.")
    lines.append("Generated from `{}`".format(records_dir))
    lines.append("")
    lines.append("Key hyperparameters and their default values "
                 "(from `configs/base.yaml`):")
    lines.append("")
    for key, info in SWEEP_PARAMS.items():
        lines.append("- `{}` = {} \u2014 {}".format(
            key, _fmt(info["default"]), info["description"]))
    lines.append("")

    has_any_variation = False

    for result in sweep_results:
        key = result["param_key"]
        info = result["param_info"]
        grouped = result["grouped"]

        lines.append("## {}".format(key))
        lines.append("")
        lines.append("*{}*".format(info.get("description", "")))
        lines.append("Default: `{}`".format(_fmt(info.get("default"))))
        lines.append("")

        if not grouped:
            lines.append("*No records found with this hyperparameter.*")
            lines.append("")
            continue

        if not result["has_variation"]:
            lines.append(
                "*All records use the same value ({}) \u2014 "
                "no sweep data available.*".format(
                    list(grouped.keys())[0]))
            lines.append("")

        lines.append("| value | n_seeds | mean acc (%) | labels |")
        lines.append("|---|---|---|---|")

        accs_by_value: Dict[Any, Optional[float]] = {}
        for value in sorted(grouped.keys(),
                            key=lambda v: (str(v) != "unknown", v)):
            g = grouped[value]
            accs_by_value[value] = g["mean_acc"]
            labels_str = ", ".join(g["labels"]) if g["labels"] else "\u2014"
            lines.append("| {} | {} | {} | {} |".format(
                _fmt(value), g["n_seeds"], _fmt(g["mean_acc"]), labels_str))
        lines.append("")

        if result["has_variation"]:
            has_any_variation = True
            valid_accs = [
                (v, a) for v, a in accs_by_value.items()
                if a is not None and v != "unknown"
            ]
            if len(valid_accs) >= 2:
                acc_values = [a for _, a in valid_accs]
                range_acc = max(acc_values) - min(acc_values)
                mean_acc = sum(acc_values) / len(acc_values)
                cv_pct = (range_acc / mean_acc * 100) if mean_acc > 0 else 0

                lines.append("**Sensitivity metrics:**")
                lines.append("")
                lines.append(
                    "- Range: {:.2f} pp (max={:.2f}%, min={:.2f}%)".format(
                        range_acc, max(acc_values), min(acc_values)))
                lines.append("- Mean: {:.2f}%".format(mean_acc))
                lines.append(
                    "- Sensitivity (range / mean): {:.1f}%".format(cv_pct))

                if range_acc < 1.0:
                    assessment = (
                        "LOW \u2014 performance is robust to this "
                        "hyperparameter")
                elif range_acc < 3.0:
                    assessment = (
                        "MODERATE \u2014 some sensitivity observed")
                else:
                    assessment = (
                        "HIGH \u2014 performance varies significantly")
                lines.append(
                    "- Assessment: **{}**".format(assessment))
                lines.append("")

    lines.append("## Summary")
    lines.append("")
    if has_any_variation:
        lines.append(
            "Sweep data detected for some hyperparameters. "
            "See per-parameter sections above.")
    else:
        lines.append(
            "*No sweep data detected in the provided records.*")
        lines.append(
            "All records use the same hyperparameter values "
            "(or values could not be extracted from `resolved_config`).")
        lines.append("")
        lines.append(
            "To generate sweep data, run the experiment with different "
            "hyperparameter values using the `--override` mechanism:")
        lines.append("")
        lines.append("```bash")
        for key, info in SWEEP_PARAMS.items():
            lines.append("# Sweep {}".format(key))
            for val in info.get("expected_values", []):
                lines.append(
                    "python runner.py --method pta --config configs "
                    "--datasets dtd --backbone ViT-B/16 "
                    "--override {}={}".format(key, val))
        lines.append("```")
    lines.append("")

    return lines


# ---------------------------------------------------------------------------
# Line plots
# ---------------------------------------------------------------------------

def generate_plots(records_dir: str, sweep_results: List[Dict[str, Any]],
                   out_path: Path) -> Optional[Path]:
    """Generate line plots (accuracy vs hyperparameter value, one line per dataset).

    Saves to ``<out_path_stem>_plots.png``.  Returns the path, or ``None``
    when matplotlib is unavailable or there is no sweep variation to plot.
    """
    if not matplotlib_available or plt is None:
        print("[WARN] matplotlib not available \u2014 skipping plots",
              file=sys.stderr)
        return None

    plot_params = [r for r in sweep_results if r["has_variation"]]
    if not plot_params:
        print("[INFO] No sweep variation detected \u2014 skipping plots")
        return None

    n = len(plot_params)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(6 * max(n, 1), 5),
                             squeeze=False)

    for idx, result in enumerate(plot_params):
        ax = axes[0][idx]
        key = result["param_key"]

        labels = list_labels(records_dir)
        ds_by_value: Dict[Any, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list))
        for label in labels:
            header, samples = load_label_records(records_dir, label)
            if not header or not samples:
                continue
            rc = header.get("resolved_config") or {}
            val = resolve_config_value(rc, key)
            if val is None:
                continue
            num_val = _to_float(val)
            norm_val: Any = round(num_val, 10) if num_val is not None else str(val)
            dataset = header.get("dataset", "unknown")
            acc = overall_acc(samples)
            if acc is not None:
                ds_by_value[norm_val][dataset].append(acc)

        all_ds = sorted(set(ds for vds in ds_by_value.values()
                            for ds in vds))
        for ds in all_ds:
            xs: List[float] = []
            ys: List[float] = []
            for val in sorted(ds_by_value.keys()):
                accs = ds_by_value[val].get(ds, [])
                if accs:
                    num = _to_float(val)
                    if num is not None:
                        xs.append(num)
                        ys.append(sum(accs) / len(accs))
            if xs:
                ax.plot(xs, ys, "o-", label=ds, linewidth=2, markersize=6)

        short = key.rsplit(".", 1)[-1]
        ax.set_xlabel(short)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy vs {}".format(key))
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = out_path.parent / (out_path.stem + "_plots.png")
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] plots saved to {}".format(plot_path))
    return plot_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hyperparameter sensitivity analysis (Exp 4.2): "
            "analyze accuracy across hyperparameter sweep values."))
    parser.add_argument(
        "--records", required=True,
        help="Record dir containing DIR/<LABEL>/records.jsonl")
    parser.add_argument(
        "--out", required=True,
        help="Output markdown file path")
    parser.add_argument(
        "--sweep", default=None,
        help=("Filter to a specific hyperparameter "
              "(e.g. image_level.alpha)"))
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating line-plot images")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    records_dir = Path(args.records)

    if not records_dir.is_dir():
        print("[ERROR] records dir not found: {}".format(records_dir),
              file=sys.stderr)
        return 1

    params_to_sweep: List[str] = []
    if args.sweep:
        if args.sweep in SWEEP_PARAMS:
            params_to_sweep = [args.sweep]
        else:
            print(
                "[WARNING] Unknown hyperparameter '{}'. "
                "Known: {}".format(
                    args.sweep, ", ".join(SWEEP_PARAMS.keys())),
                file=sys.stderr)
            params_to_sweep = [args.sweep]
            SWEEP_PARAMS[args.sweep] = {
                "description": "(custom \u2014 not in default sweep registry)",
                "default": None,
                "expected_values": [],
            }
    else:
        params_to_sweep = list(SWEEP_PARAMS.keys())

    labels = list_labels(str(records_dir))
    if not labels:
        print(
            "[WARNING] no records found in {}".format(records_dir),
            file=sys.stderr)
        print(
            "[INFO] Hyperparameter sensitivity requires sweep records "
            "with varying config values.",
            file=sys.stderr)
        print(
            "[INFO] Generating placeholder report.",
            file=sys.stderr)

    sweep_results: List[Dict[str, Any]] = []
    for param_key in params_to_sweep:
        param_info = SWEEP_PARAMS.get(param_key, {})
        result = analyze_sweep(str(records_dir), param_key, param_info)
        sweep_results.append(result)

    lines = build_report(str(records_dir), sweep_results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[OK] hyperparameter sensitivity written to {}".format(out_path))

    if not args.no_plots:
        generate_plots(str(records_dir), sweep_results, out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
