#!/usr/bin/env python3
"""Standalone validation script for Exp7–Exp12 config YAMLs.

Reads all configs_exp*/ YAML files and verifies:
1. Required keys exist per experiment type
2. caltech101 has T=50.0, others T=20.0
3. eurosat has match_threshold=0.55, others 0.60
"""

import os
import sys
import yaml

# Root of the repo
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 7 datasets for CD benchmark
DATASETS = [
    "caltech101",
    "dtd",
    "eurosat",
    "fgvc",
    "oxford_flowers",
    "oxford_pets",
    "ucf101",
]

# Shared base keys that must exist in ALL configs
SHARED_KEYS = [
    "alpha",
    "T",
    "match_threshold",
    "max_K",
    "conf_threshold",
    "conf_margin_threshold",
    "n_half",
    "soft_nn_top_m",
    "proto_alpha_max",
    "quality_eps",
    "exclude_pos",
    "patch_group_threshold",
    "gaussian_ema",
    "variance_min",
    "variance_max",
]

EXPECTED_T = {
    "caltech101": 50.0,
}

EXPECTED_MATCH_THRESHOLD = {
    "eurosat": 0.55,
}
DEFAULT_MATCH_THRESHOLD = 0.60

# Per-experiment extra keys and their expected values
EXPERIMENT_SPECS = {
    "configs_exp7": {
        "extra_keys": {"tau_text", "tau_image_min", "tau_image_max", "tau_patch_proto"},
        "forbidden_keys": {"tau_image_proto", "entropy_boost", "soft_gate_threshold", "quality_modulation"},
    },
    "configs_exp8": {
        "extra_keys": {"tau_text", "tau_image_proto", "tau_patch_proto"},
        "forbidden_keys": {"tau_image_min", "tau_image_max", "entropy_boost", "soft_gate_threshold", "quality_modulation"},
    },
    "configs_exp9": {
        "extra_keys": {"tau_text", "tau_image_proto", "tau_patch_proto", "entropy_boost"},
        "forbidden_keys": {"tau_image_min", "tau_image_max", "soft_gate_threshold", "quality_modulation"},
    },
    "configs_exp10": {
        "extra_keys": {"tau_text", "tau_image_proto", "tau_patch_proto", "soft_gate_threshold"},
        "forbidden_keys": {"tau_image_min", "tau_image_max", "entropy_boost", "quality_modulation"},
    },
    "configs_exp11": {
        "extra_keys": {"tau_text", "tau_image_proto", "tau_patch_proto"},
        "forbidden_keys": {"tau_image_min", "tau_image_max", "entropy_boost", "soft_gate_threshold", "quality_modulation"},
    },
    "configs_exp12": {
        "extra_keys": {"tau_text", "tau_image_proto", "tau_patch_proto", "quality_modulation"},
        "forbidden_keys": {"tau_image_min", "tau_image_max", "entropy_boost", "soft_gate_threshold"},
    },
}


def check_shared_params(data, dataset, filepath):
    """Verify all shared keys exist and have correct per-dataset overrides."""
    errors = []

    for key in SHARED_KEYS:
        if key not in data:
            errors.append(f"{filepath}: missing required key '{key}'")
            continue

    # Check T values
    expected_t = EXPECTED_T.get(dataset, 20.0)
    if data.get("T") != expected_t:
        errors.append(
            f"{filepath}: T={data.get('T')}, expected {expected_t}"
        )

    # Check match_threshold
    expected_mt = EXPECTED_MATCH_THRESHOLD.get(dataset, DEFAULT_MATCH_THRESHOLD)
    if data.get("match_threshold") != expected_mt:
        errors.append(
            f"{filepath}: match_threshold={data.get('match_threshold')}, expected {expected_mt}"
        )

    return errors


def validate_yaml(filepath, dataset, exp_name):
    """Validate a single YAML file."""
    errors = []

    with open(filepath, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        return [f"{filepath}: empty or invalid YAML"]

    # Check shared params
    errors.extend(check_shared_params(data, dataset, filepath))

    spec = EXPERIMENT_SPECS[exp_name]

    # Check extra keys are present
    for key in spec["extra_keys"]:
        if key not in data:
            errors.append(f"{filepath}: experiment '{exp_name}' requires key '{key}'")

    # Check forbidden keys are absent
    for key in spec["forbidden_keys"]:
        if key in data:
            errors.append(f"{filepath}: experiment '{exp_name}' should NOT have key '{key}'")

    return errors


def main():
    all_errors = []
    checked_count = 0

    for exp_name in sorted(EXPERIMENT_SPECS.keys()):
        exp_dir = os.path.join(REPO, exp_name)
        if not os.path.isdir(exp_dir):
            all_errors.append(f"Directory '{exp_dir}' does not exist")
            continue

        for dataset in DATASETS:
            filename = f"{dataset}.yaml"
            filepath = os.path.join(exp_dir, filename)
            if not os.path.isfile(filepath):
                all_errors.append(f"{filepath}: file does not exist")
                continue

            errors = validate_yaml(filepath, dataset, exp_name)
            all_errors.extend(errors)
            checked_count += 1

    expected_count = len(EXPERIMENT_SPECS) * len(DATASETS)
    print(f"Checked {checked_count}/{expected_count} config files.")

    if all_errors:
        print("\nERRORS:")
        for err in all_errors:
            print(f"  ✗ {err}")
        print("\nCONFIG CHECKS FAILED")
        sys.exit(1)
    else:
        print("\nALL CONFIG CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
