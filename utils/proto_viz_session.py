"""Session and serialization helpers for prototype-visualization tools.

This module keeps all non-UI logic needed by visualization frontends in one
place, so GUI implementations (web, desktop, notebook) can share behavior.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import os
import random
import re
from typing import Any, Dict, List, Optional

import numpy as np

from utils.proto_viz import ProtoVizEngine, SampleRecord


def _sanitize_token(value: str) -> str:
    """Return a filesystem-safe token made of lowercase letters, digits and _-."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())


def list_available_datasets(config_dir: str) -> List[str]:
    """List dataset names inferred from YAML files in a config directory.

    Args:
        config_dir: Directory containing per-dataset YAML config files.

    Returns:
        Sorted dataset names (file stem without the `.yaml` suffix).
    """
    if not os.path.isdir(config_dir):
        return []
    datasets = []
    for name in os.listdir(config_dir):
        if name.endswith(".yaml"):
            datasets.append(os.path.splitext(name)[0])
    return sorted(datasets)


def sample_record_to_dict(rec: SampleRecord, selected_class_name: Optional[str] = None) -> Dict[str, Any]:
    """Convert a `SampleRecord` into a JSON-serializable dictionary.

    Args:
        rec: One processed sample snapshot.
        selected_class_name: Optional class name for focused bank stats.

    Returns:
        Dictionary with top-level summary, class table, match table, and selected
        class bank details. Numpy arrays are omitted by design to keep payloads
        compact for web clients.
    """
    sorted_classes = sorted(rec.classes, key=lambda c: -c.final_logit)
    classes = []
    for cr in sorted_classes:
        classes.append({
            "class_id": cr.class_id,
            "class_name": cr.class_name,
            "bank_K": cr.bank_size,
            "max_k": int(getattr(rec, "update_gate_max_k", 0)),
            "text_score": cr.text_score,
            "raw_proto": cr.raw_proto_score,
            "delta_proto": getattr(cr, "delta_proto", cr.raw_proto_score),
            "alpha": getattr(cr, "alpha", cr.class_penalty),
            "quality_gate": getattr(cr, "quality_gate", 0.0),
            "tau_proto": getattr(cr, "tau_proto", 0.0),
            "centered": cr.centered_proto,
            "proto_term": cr.proto_term,
            "tau_eff": cr.tau_eff,
            "class_penalty": cr.class_penalty,
            "final_logit": cr.final_logit,
            "prob": cr.softmax_prob,
        })

    all_matches = getattr(rec, "all_matches", rec.top_matches)

    foreground_thresh = float(getattr(rec, "foreground_appw_thresh", 0.5))

    matches = []
    for m in rec.top_matches[:20]:
        update_samples = int(getattr(m.proto, "update_samples", 0))
        app_w = float(m.proto.appearance) / update_samples if update_samples > 0 else 0.0
        matches.append({
            "class_id": m.proto.class_id,
            "class_name": m.proto.class_name,
            "proto_idx": m.proto.proto_idx,
            "appearance": m.proto.appearance,
            "app_w": app_w,
            "update_samples": update_samples,
            "is_foreground": app_w >= foreground_thresh,
            "evidence_weight": m.proto.evidence_weight,
            "max_sim": m.max_patch_sim,
            "centered_score": m.centered_score,
            "contrib": m.final_contrib,
            "top_patch_indices": list(m.top_patch_indices),
            "proto_patch_data": m.proto.rep_patch_data,
            "proto_patch_idx": m.proto.rep_patch_idx,
            "proto_patch_sim": m.proto.rep_patch_sim,
        })

    class_prototypes: Dict[str, List[Dict[str, Any]]] = {}
    for m in all_matches:
        update_samples = int(getattr(m.proto, "update_samples", 0))
        app_w = float(m.proto.appearance) / update_samples if update_samples > 0 else 0.0
        key = str(m.proto.class_id)
        class_prototypes.setdefault(key, []).append({
            "proto_idx": m.proto.proto_idx,
            "appearance": m.proto.appearance,
            "app_w": app_w,
            "update_samples": update_samples,
            "is_foreground": app_w >= foreground_thresh,
            "evidence_weight": m.proto.evidence_weight,
            "max_sim": m.max_patch_sim,
            "centered_score": m.centered_score,
            "contrib": m.final_contrib,
            "top_patch_indices": list(m.top_patch_indices),
            "proto_patch_data": m.proto.rep_patch_data,
            "proto_patch_idx": m.proto.rep_patch_idx,
            "proto_patch_sim": m.proto.rep_patch_sim,
            "class_id": m.proto.class_id,
            "class_name": m.proto.class_name,
        })

    for key in class_prototypes:
        class_prototypes[key].sort(key=lambda x: x["proto_idx"])

    selected = selected_class_name or rec.target_name
    selected_class = next((c for c in rec.classes if c.class_name == selected), None)
    if selected_class is None:
        selected_class = next((c for c in rec.classes if c.class_id == rec.target), None)

    selected_matches = []
    if selected_class is not None:
        selected_matches = [
            {
                "proto_idx": m.proto.proto_idx,
                "appearance": m.proto.appearance,
                "app_w": (float(m.proto.appearance) / int(getattr(m.proto, "update_samples", 0)))
                         if int(getattr(m.proto, "update_samples", 0)) > 0 else 0.0,
                "evidence_weight": m.proto.evidence_weight,
                "max_sim": m.max_patch_sim,
                "centered_score": m.centered_score,
                "contrib": m.final_contrib,
            }
            for m in all_matches
            if m.proto.class_id == selected_class.class_id
        ]
        selected_matches.sort(key=lambda x: x["proto_idx"])

    return {
        "sample_idx": rec.sample_idx,
        "target": rec.target,
        "target_name": rec.target_name,
        "predicted": rec.predicted,
        "predicted_name": rec.predicted_name,
        "correct": rec.correct,
        "clip_pred": rec.clip_pred,
        "clip_conf": rec.clip_conf,
        "running_acc": rec.running_acc,
        "text_running_acc": float(getattr(rec, "text_running_acc", 0.0)),
        "update_gate": {
            "passed": bool(getattr(rec, "update_gate_passed", False)),
            "class_id": getattr(rec, "update_gate_class_id", None),
            "class_name": getattr(rec, "update_gate_class_name", None),
            "best_conf": float(getattr(rec, "update_gate_best_conf", 0.0)),
            "second_conf": float(getattr(rec, "update_gate_second_conf", 0.0)),
            "margin": float(getattr(rec, "update_gate_margin", 0.0)),
            "conf_thresh": float(getattr(rec, "update_gate_conf_thresh", 0.0)),
            "margin_thresh": float(getattr(rec, "update_gate_margin_thresh", 0.0)),
            "reason": str(getattr(rec, "update_gate_reason", "")),
            "bank_k": int(getattr(rec, "update_gate_bank_k", 0)),
            "max_k": int(getattr(rec, "update_gate_max_k", 0)),
        },
        "proto_formula": {
            "tau_proto": float(getattr(rec, "tau_proto", 0.0)),
            "quality_gate": float(getattr(rec, "quality_gate", 0.0)),
            "foreground_appw_thresh": foreground_thresh,
        },
        "num_samples": None,
        "image_data": rec.image_data,
        "classes": classes,
        "top_matches": matches,
        "class_prototypes": class_prototypes,
        "selected_class_name": selected_class.class_name if selected_class else None,
        "selected_class": {
            "class_id": selected_class.class_id,
            "class_name": selected_class.class_name,
            "bank_K": selected_class.bank_size,
            "text_score": selected_class.text_score,
            "raw_proto": selected_class.raw_proto_score,
            "centered": selected_class.centered_proto,
            "proto_term": selected_class.proto_term,
            "tau_eff": selected_class.tau_eff,
            "class_penalty": selected_class.class_penalty,
            "final_logit": selected_class.final_logit,
            "prob": selected_class.softmax_prob,
            "matches": selected_matches,
        } if selected_class else None,
    }


class ProtoVizSession:
    """In-memory state for navigating visualization records.

    The session encapsulates dataset metadata, current index, and helper actions
    like next/restart/export. Frontends can call this API without handling model
    internals or record serialization details.
    """

    def __init__(self) -> None:
        self.dataset_name: str = ""
        self.source: str = ""
        self.records: List[SampleRecord] = []
        self.classnames: List[str] = []
        self.current_idx: int = 0

    @property
    def loaded(self) -> bool:
        """Whether a dataset/record set is currently active."""
        return len(self.records) > 0

    def load_replay(self, records_path: str, dataset_name: Optional[str] = None) -> None:
        """Load pre-saved records from disk and reset navigation state."""
        records = ProtoVizEngine.load_records(records_path)
        if not records:
            raise ValueError(f"No records found in {records_path}")

        self.records = records
        self.classnames = [c.class_name for c in records[0].classes]
        if dataset_name:
            self.dataset_name = dataset_name
        else:
            stem = os.path.splitext(os.path.basename(records_path))[0]
            self.dataset_name = stem.split("_n", 1)[0]
        self.source = records_path
        self.current_idx = 0

    def load_live(
        self,
        dataset: str,
        config: str,
        backbone: str,
        data_root: str,
        n_samples: int,
    ) -> str:
        """Run live processing and store generated records in the session.

        Returns:
            Path to the auto-saved record pickle file.
        """
        import clip as clip_lib
        import torch

        from utils import build_test_data_loader, clip_classifier, get_config_file

        clip_model, preprocess = clip_lib.load(backbone)
        clip_model.eval()

        cfg = get_config_file(config, dataset)

        # Make sample order and stochastic transforms deterministic across app restarts.
        viz_seed = int(cfg.get("viz_seed", 1))
        random.seed(viz_seed)
        np.random.seed(viz_seed)
        torch.manual_seed(viz_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(viz_seed)

        loader, classnames, template = build_test_data_loader(
            dataset,
            data_root,
            preprocess,
            shuffle=False,
        )
        clip_weights = clip_classifier(classnames, template, clip_model)
        engine = ProtoVizEngine(cfg, classnames, clip_weights, clip_model)

        records: List[SampleRecord] = []
        with torch.no_grad():
            for i, (images, targets) in enumerate(loader):
                if i >= n_samples:
                    break
                records.append(engine.run_sample(images, targets))

        os.makedirs("outputs/viz", exist_ok=True)
        save_path = f"outputs/viz/{dataset}_n{n_samples}.pkl"
        engine.save_records(records, save_path)

        self.records = records
        self.classnames = classnames
        self.dataset_name = dataset
        self.source = save_path
        self.current_idx = 0
        return save_path

    def current_payload(self, selected_class_name: Optional[str] = None) -> Dict[str, Any]:
        """Return serialized payload for the current sample pointer."""
        if not self.loaded:
            raise RuntimeError("No records loaded")
        rec = self.records[self.current_idx]
        payload = sample_record_to_dict(rec, selected_class_name=selected_class_name)
        payload["num_samples"] = len(self.records)
        payload["dataset"] = self.dataset_name
        payload["classnames"] = self.classnames
        return payload

    def next(self) -> int:
        """Advance pointer by one sample, clamped at the last sample index."""
        if not self.loaded:
            raise RuntimeError("No records loaded")
        if self.current_idx < len(self.records) - 1:
            self.current_idx += 1
        return self.current_idx

    def restart(self) -> int:
        """Move pointer to the first sample and return the new index (0)."""
        if not self.loaded:
            raise RuntimeError("No records loaded")
        self.current_idx = 0
        return self.current_idx

    def set_index(self, idx: int) -> int:
        """Set pointer to an explicit sample index (with bounds checking)."""
        if not self.loaded:
            raise RuntimeError("No records loaded")
        if idx < 0 or idx >= len(self.records):
            raise IndexError(f"Index {idx} out of range for {len(self.records)} samples")
        self.current_idx = idx
        return self.current_idx

    def fast_forward_to(self, idx: int) -> int:
        """Move to idx using next-step semantics; restart if idx is behind current."""
        if not self.loaded:
            raise RuntimeError("No records loaded")
        if idx < 0 or idx >= len(self.records):
            raise IndexError(f"Index {idx} out of range for {len(self.records)} samples")

        if idx < self.current_idx:
            self.restart()

        while self.current_idx < idx:
            self.next()

        return self.current_idx

    def export_current(self, output_dir: str = "outputs/viz_exports") -> str:
        """Save current sample details to a timestamped JSON file.

        Filename pattern:
            YYYYmmdd_HHMMSS_<dataset>_sample<idx>.json
        """
        if not self.loaded:
            raise RuntimeError("No records loaded")

        rec = self.records[self.current_idx]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_token = _sanitize_token(self.dataset_name or "dataset")
        sample_token = f"sample{rec.sample_idx:05d}"
        file_name = f"{ts}_{dataset_token}_{sample_token}.json"

        payload = {
            "metadata": {
                "dataset": self.dataset_name,
                "source": self.source,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            },
            "sample": sample_record_to_dict(rec),
        }

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, file_name)
        with open(out_path, "w", encoding="utf-8") as f:
            import json
            json.dump(payload, f, indent=2)

        return out_path
