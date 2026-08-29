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
* ``write_convergence(batch_correct)`` — per-run convergence/online-accuracy
  trajectory (``convergence.json``); used by the Part 4b two-sided down-weight
  study to measure how fast adaptation converges.

Convergence JSON schema (batch_size=1 ⇒ one correctness value per sample):
* ``total``: int — number of samples
* ``final_acc``: float — overall accuracy (%)
* ``online_acc``: [float] — cumulative-accuracy trajectory at every sample n
  (100 * (#correct among first n) / n), the primary convergence curve
* ``acc_at_fractions``: {str: float} — accuracy reached after 10/25/50/75/100%
  of the stream seen (``"0.1"``, ``"0.25"``, ``"0.5"``, ``"0.75"``, ``"1.0"``)
* ``t_to_80pct``: int|null — first sample index (1-based) where
  ``online_acc >= 0.8 * final_acc``, a time-to-reach-target measure
* ``auc_norm``: float — mean of normalized online accuracy
  (``online_acc / final_acc``), a single-scalar speed-vs-plateau summary
  (> 1 ⇨ reaches above its plateau early; < 1 ⇨ lags)
"""
import datetime
import json
import os
from typing import Any, List, Optional

__all__ = ["write_record_header", "write_record", "write_summary", "write_convergence"]

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
    write_gate: Optional[str] = None,
    write_occurred: Optional[bool] = None,
    patch_vote_pred: Optional[int] = None,
    patch_vote_margin: Optional[float] = None,
    clip_margin: Optional[float] = None,
    boost: Optional[float] = None,
    boost_down: Optional[float] = None,
) -> None:
    """Append one per-sample record to ``records.jsonl``.

    Missing/unknown fields should be passed as ``None`` — they serialize as
    JSON ``null`` (never empty lists). No-op when ``RECORD_DIR`` is unset.

    The optional ``write_gate`` / ``write_occurred`` / ``patch_vote_pred`` /
    ``patch_vote_margin`` / ``clip_margin`` fields are used by the
    write-gate PTA study (``models/write_gate_pta.py``) and the write-reweight
    study (``models/reweight_pta.py``, which additionally records the applied
    ``boost``) to record write-rule diagnostics; all default to ``None`` and
    are strictly backward-compatible with existing callers (PTA / PatchModPTA /
    ZeroShot). The two-sided Part 4b down-weight study also records
    ``boost_down`` (the down-weight factor config used for untrusted writes).
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
        "write_gate": write_gate,
        "write_occurred": write_occurred,
        "patch_vote_pred": patch_vote_pred,
        "patch_vote_margin": patch_vote_margin,
        "clip_margin": clip_margin,
        "boost": boost,
        "boost_down": boost_down,
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


def write_convergence(batch_correct) -> None:
    """Write ``convergence.json`` with the online-accuracy trajectory.

    ``batch_correct`` is an iterable of per-sample correctness values (batch
    size is 1, so one value per sample — accept bools or 0/1 numerics). This is
    the primary evidence for the Part 4b down-weight study's convergence-speed
    hypothesis: whether re-weighting trusted vs untrusted writes changes how
    *fast* adaptation reaches its plateau, not just where it lands.

    No-op when ``RECORD_DIR`` is unset or empty (mirrors the other writers).
    """
    record_dir = _record_dir()
    if not record_dir:
        return
    _ensure_dir(record_dir)

    corrects = [1.0 if c else 0.0 for c in batch_correct]
    n = len(corrects)
    if n == 0:
        payload = _json_safe({
            "total": 0, "final_acc": 0.0, "online_acc": [],
            "acc_at_fractions": {}, "t_to_80pct": None, "auc_norm": None,
        })
        with open(os.path.join(record_dir, "convergence.json"), "w") as f:
            json.dump(payload, f)
        return

    running = 0.0
    online_acc = []
    for c in corrects:
        running += c
        online_acc.append(100.0 * running / len(online_acc + [1.0]))

    final_acc = online_acc[-1]

    def acc_at_fraction(frac):
        idx = max(1, int(round(n * frac))) - 1
        return round(online_acc[idx], 3)

    acc_at_fractions = {
        "0.1": acc_at_fraction(0.1),
        "0.25": acc_at_fraction(0.25),
        "0.5": acc_at_fraction(0.5),
        "0.75": acc_at_fraction(0.75),
        "1.0": round(final_acc, 3),
    }

    t_to_80pct = None
    if final_acc > 1e-6:
        for idx, a in enumerate(online_acc):
            if a >= 0.8 * final_acc:
                t_to_80pct = idx + 1
                break

    auc_norm = None
    if final_acc > 1e-6:
        auc_norm = round(sum(a / final_acc for a in online_acc) / n, 4)

    payload = _json_safe({
        "total": n,
        "final_acc": round(final_acc, 3),
        "online_acc": [round(a, 3) for a in online_acc],
        "acc_at_fractions": acc_at_fractions,
        "t_to_80pct": t_to_80pct,
        "auc_norm": auc_norm,
    })
    with open(os.path.join(record_dir, "convergence.json"), "w") as f:
        json.dump(payload, f)
