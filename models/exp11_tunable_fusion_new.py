"""
Experiment 11: Tunable Fusion New
====================================

Identical algorithm to Exp5 (Tunable Fusion with configurable tau weights),
but uses a separate config directory `configs_exp11/` with its own
hyperparameters (different tau values, quality thresholds, etc.).

This module re-exports the build() factory from exp5_tunable_fusion so the
runner can import it as models.exp11_tunable_fusion_new (proper Exp11 naming).
"""

from models.exp5_tunable_fusion import Exp5TunableFusionAdapter, build

__all__ = ["Exp5TunableFusionAdapter", "build"]
