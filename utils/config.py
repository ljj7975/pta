"""Configuration loading with OmegaConf YAML inheritance chains."""

import os
from omegaconf import OmegaConf, DictConfig


def _resolve_config_chain(config_file: str) -> dict:
    """Load a YAML config and recursively resolve its ``defaults`` list.

    Each entry in ``defaults`` is a path (relative to the current file's
    directory, without the ``.yaml`` extension) to a parent config.
    Parents are merged in order, with the current file's values taking
    precedence.  Returns a plain dict.
    """
    cfg: DictConfig = OmegaConf.load(config_file)

    if "defaults" not in cfg:
        OmegaConf.resolve(cfg)
        return dict(OmegaConf.to_container(cfg))  # type: ignore[arg-type]

    defaults: list = list(cfg.pop("defaults"))
    config_dir = os.path.dirname(os.path.abspath(config_file))

    parents: list[DictConfig] = []
    for ref in defaults:
        parent_path = os.path.normpath(os.path.join(config_dir, ref + ".yaml"))
        parent_dict = _resolve_config_chain(parent_path)
        parents.append(OmegaConf.create(parent_dict))

    merged = OmegaConf.merge(*parents, cfg)
    OmegaConf.resolve(merged)
    return dict(OmegaConf.to_container(merged))  # type: ignore[arg-type]


def get_config_file(config_path: str, dataset_name: str) -> dict:
    """Resolve a dataset name to its YAML config and return the merged dict.

    If *config_path* is an absolute path to a ``.yaml`` file it is used
    directly; otherwise the function looks for
    ``<config_path>/<dataset>.yaml`` (with special handling for ImageNet
    variants ``I / A / V / R / S``).
    """
    if config_path.endswith(".yaml") and os.path.isfile(config_path):
        config_file = config_path
    else:
        if dataset_name == "I":
            config_name = "imagenet.yaml"
        elif dataset_name in ["A", "V", "R", "S"]:
            config_name = f"imagenet_{dataset_name.lower()}.yaml"
        else:
            config_name = f"{dataset_name}.yaml"
        config_file = os.path.join(config_path, config_name)

    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"The configuration file {config_file} was not found."
        )

    return _resolve_config_chain(config_file)
