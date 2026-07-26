from models.image_level.base import BaseImageLevel
from models.image_level.pta_image import PTAImageLevel, build


def create(cfg: dict) -> BaseImageLevel:
    """Create default image-level component (PTAImageLevel) from config.

    Handles flat (legacy) config format by wrapping into the nested
    ``{"image_level": ...}`` structure expected by the component.
    """
    if "image_level" not in cfg:
        cfg = {"image_level": dict(cfg)}
    return PTAImageLevel(cfg)


__all__ = ["BaseImageLevel", "PTAImageLevel", "build", "create"]
