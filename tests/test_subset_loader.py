#!/usr/bin/env python3
"""
Tests for the closed-set build_subset_test_data_loader (utils/data.py).

Run with: python tests/test_subset_loader.py

Covers (on a 2-class dtd subset, ~72 images):
  a) loader yields only remapped labels in {0, 1}
  b) len(loader) is 72 (36 img/class x 2)
  c) unknown class name raises ValueError
  d) class_file parsing (incl. comment lines)
  e) subset classnames returned in input order
"""
import os
import sys
import tempfile
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import build_subset_test_data_loader

DATA_ROOT = "./data"
DATASET = "dtd"

# CLIP-free preprocess (DatasetWrapper applies tfm to a PIL image).
PREPROCESS = transforms.Compose(
    [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


def test_two_class_remap():
    loader, classnames, template = build_subset_test_data_loader(
        DATASET, DATA_ROOT, PREPROCESS, class_names=["banded", "blotchy"]
    )
    assert len(classnames) == 2, f"expected 2 classnames, got {len(classnames)}"
    seen = set()
    for images, labels in loader:
        for label in labels.tolist():
            assert label in (0, 1), f"label {label} not remapped into {{0, 1}}"
            seen.add(label)
    assert seen == {0, 1}, f"expected labels {{0, 1}}, got {seen}"
    assert len(loader) == 72, f"expected 72 batches (36/class x 2), got {len(loader)}"


def test_unknown_class_raises():
    try:
        build_subset_test_data_loader(
            DATASET, DATA_ROOT, PREPROCESS, class_names=["not_a_real_class"]
        )
    except ValueError as e:
        assert "Unknown class name" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown class name")


def test_class_file_parsing():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("# difficult classes for DTD\n")
        f.write("\n")
        f.write("banded\n")
        f.write("blotchy\n")
        f.write("  # trailing comment line\n")
        path = f.name
    try:
        loader, classnames, template = build_subset_test_data_loader(
            DATASET, DATA_ROOT, PREPROCESS, class_file=path
        )
        assert classnames == ["banded", "blotchy"], f"got {classnames}"
        assert len(loader) == 72, f"expected 72 batches, got {len(loader)}"
    finally:
        os.unlink(path)


def test_classnames_in_input_order():
    loader, classnames, template = build_subset_test_data_loader(
        DATASET, DATA_ROOT, PREPROCESS, class_names=["blotchy", "banded"]
    )
    assert classnames == ["blotchy", "banded"], f"order not preserved: {classnames}"
    seen = set()
    for images, labels in loader:
        for label in labels.tolist():
            seen.add(label)
    # Input order dictates remap: 'blotchy' -> 0, 'banded' -> 1.
    assert seen == {0, 1}


if __name__ == "__main__":
    test_two_class_remap()
    print("subset_loader: PASS two-class remap (labels in {0,1}, len==72)")
    test_unknown_class_raises()
    print("subset_loader: PASS unknown class raises ValueError")
    test_class_file_parsing()
    print("subset_loader: PASS class_file parsing (comments, len==72)")
    test_classnames_in_input_order()
    print("subset_loader: PASS classnames in input order")
    print("subset_loader: ALL TESTS PASSED")
