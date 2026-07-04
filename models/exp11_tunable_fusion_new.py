"""
Experiment 11: Tunable Fusion New
====================================

Identical algorithm to Exp5 (Tunable Fusion with configurable tau weights),
but uses a separate config directory `configs_exp11/` with its own
hyperparameters (different tau values, quality thresholds, etc.).

Subclasses Exp5TunableFusionAdapter and overrides EXP_LABEL so results
are written with the correct "Exp11TunableFusionNew" label.
"""
from models.exp5_tunable_fusion import Exp5TunableFusionAdapter


class Exp11TunableFusionNewAdapter(Exp5TunableFusionAdapter):
    """Same algorithm as Exp5, different config -> different label."""

    EXP_LABEL = "Exp11TunableFusionNew"


def build(cfg: dict) -> Exp11TunableFusionNewAdapter:
    """Factory function — called by runner.py via dynamic import."""
    return Exp11TunableFusionNewAdapter(cfg)


__all__ = ["Exp11TunableFusionNewAdapter", "build"]
