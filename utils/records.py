"""Env-gated per-sample JSONL record writer + per-class summary writer.

Pure I/O utility shared by the PatchModPTA / PTA / ZeroShot adapters for
behavior-neutral per-sample recording (see plan
``.omo/plans/perclass-difficult-classes-validation.md``, Task 1).

Gating
------
Every function reads the ``RECORD_DIR`` environment variable on each call.
When it is unset or empty, all functions are strict no-ops: no files are
created, no state is touched, nothing is printed. When set, per-sample
records are appended to ``$RECORD_DIR/records.jsonl`` (one JSON object per
line; the first line is the ``__header__`` object) and :func:`write_summary`
writes ``$RECORD_DIR/summary.json``. The record directory is created with
``os.makedirs(..., exist_ok=True)``, so nested paths work.

Constraints
-----------
* No torch / numpy imports — tensor-like inputs are duck-typed and converted
  by :func:`_json_safe` (fp16 is cast to float32 before ``json.dumps``).
* No RNG, no prints, no mutation of caller state.
* ``None`` fields serialize as JSON ``null`` — never as empty lists.

Schema (field names are load-bearing — ``scripts/analyze_records.py``, T4,
consumes this exact schema):

* Header line: ``{"__header__": true, "method": str, "dataset": str,
  "seed": int, "C": int, "classnames": [str], "resolved_config": {...},
  "created_at": iso-str}``
* Per-sample: ``{"batch_idx": int, "target": int, "pred": int,
  "correct": bool, "conf": float, "quality_gate": float|null,
  "gate_mode": "single"|"multi"|null, "proto_alpha": [float]|null,
  "logits": {"clip": [float]|null, "image_proto": [float]|null,
  "patch_proto": [float]|null, "final": [float]},
  "proto_stats": {"true": {...}|null, "pred": {...}|null}}``
* Summary: ``{"method", "dataset", "seed", "total": int, "acc": float,
  "per_class": {"<classname>": {"total": int, "correct": int, "acc": float,
  "n_images_end": int, "n_clusters_end": int}}}`` (``n_*`` optional per class).

API
---
* ``write_record_header(method, dataset, seed, C, classnames, resolved_config)``
* ``write_record(batch_idx, target, pred, correct, conf, quality_gate=None,
  gate_mode=None, proto_alpha=None, logits=None, proto_stats=None)``
* ``write_summary(method, dataset, seed, total, acc, per_class)``
"""
import datetime
import json
import os
from typing import Any, List, Optional

__all__ = ["write_record_header", "write_record", "write_summary"]

_RECORD_FILE = "records.jsonl"
_SUMMARY_FILE = "summary.json"


def _record_dir() -> str:
    """Active record directory, or '' when gating is off (no-op mode)."""
    return os.environ.get("RECORD_DIR", "").strip()


def _ensure_dir(record_dir: str) -> None:
    os.makedirs(record_dir, exist_ok=True)


def _json_safe(obj: Any) -> Any:
    """Recursively convert tensor/numpy values to JSON-native Python objects.

    Duck-typed so this module never imports torch or numpy:
    * torch tensor (has ``cpu`` + ``tolist``): fp16 → ``.float()`` first, then
      ``.cpu().tolist()``; 0-dim tensors go through ``.item()`` (spec: scalar
      tensors → ``.item()``).
    * numpy scalar / array: ``.item()`` when scalar, ``.tolist()`` otherwise.
    * ``None`` stays ``None`` (JSON ``null``); dicts/lists recurse.
    """
    if obj is None or isinstance(obj, (bool, str, int, float)):
        return obj
    if hasattr(obj, "tolist") and hasattr(obj, "cpu"):
        # torch tensor
        if hasattr(obj, "dtype") and "float16" in str(obj.dtype):
            obj = obj.float()
        obj = obj.cpu()
        if getattr(obj, "ndim", 1) == 0 and hasattr(obj, "item"):
            return obj.item()
        return obj.tolist()
    if hasattr(obj, "item") and hasattr(obj, "dtype"):
        # numpy scalar (or array with a single element)
        try:
            return obj.item()
        except ValueError:
            return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def write_record_header(
    method: str,
    dataset: str,
    seed: int,
    C: int,
    classnames: List[str],
    resolved_config: Any,
) -> None:
    """Write the ``__header__`` line of ``records.jsonl``.

    No-op when ``RECORD_DIR`` is unset or empty.
    """
    record_dir = _record_dir()
    if not record_dir:
        return
    _ensure_dir(record_dir)
    payload = _json_safe({
        "__header__": True,
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "C": C,
        "classnames": classnames,
        "resolved_config": resolved_config,
        "created_at": datetime.datetime.now().isoformat(),
    })
    with open(os.path.join(record_dir, _RECORD_FILE), "a") as f:
        _ = f.write(json.dumps(payload) + "\n")


def write_record(
    batch_idx: int,
    target: int,
    pred: int,
    correct: bool,
    conf: float,
    quality_gate: Optional[Any] = None,
    gate_mode: Optional[str] = None,
    proto_alpha: Optional[Any] = None,
    logits: Optional[Any] = None,
    proto_stats: Optional[Any] = None,
) -> None:
    """Append one per-sample record to ``records.jsonl``.

    Missing/unknown fields should be passed as ``None`` — they serialize as
    JSON ``null`` (never empty lists). No-op when ``RECORD_DIR`` is unset.
    """
    record_dir = _record_dir()
    if not record_dir:
        return
    _ensure_dir(record_dir)
    payload = _json_safe({
        "batch_idx": batch_idx,
        "target": target,
        "pred": pred,
        "correct": correct,
        "conf": conf,
        "quality_gate": quality_gate,
        "gate_mode": gate_mode,
        "proto_alpha": proto_alpha,
        "logits": logits,
        "proto_stats": proto_stats,
    })
    with open(os.path.join(record_dir, _RECORD_FILE), "a") as f:
        _ = f.write(json.dumps(payload) + "\n")


def write_summary(
    method: str,
    dataset: str,
    seed: int,
    total: int,
    acc: float,
    per_class: Any,
) -> None:
    """Write ``summary.json`` with the per-class accuracy breakdown.

    ``per_class`` maps classname → ``{"total", "correct", "acc", [optional
    "n_images_end", "n_clusters_end"]}``. No-op when ``RECORD_DIR`` is unset.
    """
    record_dir = _record_dir()
    if not record_dir:
        return
    _ensure_dir(record_dir)
    payload = _json_safe({
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "total": total,
        "acc": acc,
        "per_class": per_class,
    })
    with open(os.path.join(record_dir, _SUMMARY_FILE), "w") as f:
        json.dump(payload, f)
