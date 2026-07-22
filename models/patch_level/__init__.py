from models.patch_level.base import BasePatchLevel, _alpha_from_evidence
from models.patch_level.gaussian_patch import GaussianPatchLevel, build
from utils.kmeans import _incremental_kmeans_step, _gaussian_score_for_class
from utils.clip_inference import _safe_normalize


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
    "_incremental_kmeans_step",
    "_alpha_from_evidence",
    "_gaussian_score_for_class",
]
