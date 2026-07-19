from models.patch_level.base import (
    BasePatchLevel,
    _safe_normalize,
    _extract_patch_embeddings,
    _extract_all_tokens,
    _incremental_kmeans_step,
    _alpha_from_evidence,
    _gaussian_score_for_class,
)
from models.patch_level.gaussian_patch import GaussianPatchLevel, build


def create(cfg: dict) -> BasePatchLevel:
    """Create default patch-level component (GaussianPatchLevel) from config.

    Handles flat (legacy) config format by wrapping into the nested
    ``{"patch_level": ...}`` structure expected by the component.
    """
    if "patch_level" not in cfg:
        cfg = {"patch_level": dict(cfg)}
    return GaussianPatchLevel(cfg)


__all__ = [
    "BasePatchLevel",
    "GaussianPatchLevel",
    "build",
    "create",
    "_safe_normalize",
    "_extract_patch_embeddings",
    "_extract_all_tokens",
    "_incremental_kmeans_step",
    "_alpha_from_evidence",
    "_gaussian_score_for_class",
]
