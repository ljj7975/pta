"""Helpers for ImageNet subset remapping (used by A/R/S variants)."""

import os
import torch
from datasets.imagenet import ImageNet


def get_imagenet_subset_remap(
    dataset_name: str, root_path: str, subset_classnames: list
):
    """Return label-remapping tensors for ImageNet-A/R/S evaluation.

    These datasets are 200-class subsets of ImageNet.  The paper evaluates
    with 1000-class CLIP weights and then remaps predictions to the subset
    label space.

    Returns:
        imagenet_classnames: full 1000-class name list (ordered by imagenet label).
        subset_to_imagenet: LongTensor[200] mapping subset-label \u2192 imagenet-label.
        imagenet_to_subset: dict mapping imagenet-label \u2192 subset-label (-1 if absent).
    """
    imagenet_classnames_dict = ImageNet.read_classnames(
        os.path.join(root_path, "imagenet", "classnames.txt")
    )
    imagenet_synsets = list(imagenet_classnames_dict.keys())
    imagenet_classnames = [imagenet_classnames_dict[s] for s in imagenet_synsets]
    imagenet_idx = {s: i for i, s in enumerate(imagenet_synsets)}

    dataset_dir_map = {
        "A": os.path.join(root_path, "imagenet-adversarial", "imagenet-a"),
        "R": os.path.join(root_path, "imagenet-rendition", "imagenet-r"),
        "S": os.path.join(root_path, "imagenet-sketch", "images"),
    }
    from datasets.utils import listdir_nohidden

    image_dir = dataset_dir_map[dataset_name]
    folders = sorted(
        [
            f
            for f in listdir_nohidden(image_dir, sort=True)
            if f not in ["README.txt"]
        ]
    )

    subset_to_imagenet = []
    for folder in folders:
        if folder in imagenet_idx:
            subset_to_imagenet.append(imagenet_idx[folder])
        else:
            subset_to_imagenet.append(-1)

    subset_to_imagenet = torch.tensor(subset_to_imagenet, dtype=torch.long)
    imagenet_to_subset = {
        im_lbl: sub_lbl
        for sub_lbl, im_lbl in enumerate(subset_to_imagenet.tolist())
        if im_lbl >= 0
    }

    return imagenet_classnames, subset_to_imagenet, imagenet_to_subset
