"""
Utils package — re-exports all public symbols for backward compatibility.

Module layout:
    config.py            _resolve_config_chain, get_config_file
    data.py              worker_init_fn, get_ood_preprocess, build_test_data_loader, _LabelRemapLoader
    clip_inference.py    clip_classifier, get_clip_logits
    metrics.py           cls_acc, select_confident_samples, softmax_entropy, avg_entropy
    dataset_helpers.py   get_imagenet_subset_remap
    timer.py             Timer
    proto_viz.py         (visualization utilities)
    proto_viz_session.py
"""

# -- Config loading --
from .config import _resolve_config_chain, get_config_file

# -- Data loaders & transforms --
from .data import (
    BICUBIC,
    _LabelRemapLoader,
    build_test_data_loader,
    get_ood_preprocess,
    worker_init_fn,
)

# -- CLIP inference --
from .clip_inference import clip_classifier, get_clip_logits

# -- Metrics & entropy --
from .metrics import (
    avg_entropy,
    cls_acc,
    select_confident_samples,
    softmax_entropy,
)

# -- Dataset helpers --
from .dataset_helpers import get_imagenet_subset_remap

__all__ = [
    # config
    "_resolve_config_chain",
    "get_config_file",
    # data
    "BICUBIC",
    "_LabelRemapLoader",
    "build_test_data_loader",
    "get_ood_preprocess",
    "worker_init_fn",
    # clip inference
    "clip_classifier",
    "get_clip_logits",
    # metrics
    "avg_entropy",
    "cls_acc",
    "select_confident_samples",
    "softmax_entropy",
    # dataset helpers
    "get_imagenet_subset_remap",
]
