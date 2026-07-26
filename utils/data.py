"""Data loaders, transforms, and worker utilities."""

import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader, AugMixAugmenter

_BICUBIC = InterpolationMode.BICUBIC


def _worker_init_fn(worker_id: int):
    """Seed each DataLoader worker for deterministic behavior with num_workers > 0."""
    import random
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _get_ood_preprocess():
    """Build an AugMix preprocessing pipeline for OOD ImageNet variants."""
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    base_transform = transforms.Compose(
        [transforms.Resize(224, interpolation=_BICUBIC), transforms.CenterCrop(224)]
    )
    preprocess = transforms.Compose([transforms.ToTensor(), normalize])
    aug_preprocess = AugMixAugmenter(
        base_transform, preprocess, n_views=63, augmix=True
    )
    return aug_preprocess


class _LabelRemapLoader:
    """Wraps a DataLoader and remaps 200-class subset labels to 1000-class imagenet labels."""

    def __init__(self, loader, subset_to_imagenet):
        self._loader = loader
        self._subset_to_imagenet = subset_to_imagenet

    def __iter__(self):
        for images, targets in self._loader:
            remapped = self._subset_to_imagenet[targets]
            yield images, remapped

    def __len__(self):
        return len(self._loader)


def build_test_data_loader(dataset_name: str, root_path: str, preprocess, shuffle: bool = True):
    """Build a test DataLoader for the given dataset.

    Returns ``(test_loader, classnames, template)``.

    For ImageNet-A/R/S the loader is wrapped with ``_LabelRemapLoader`` so that
    200-class subset labels are remapped to 1000-class ImageNet labels at
    iteration time (matching the paper evaluation protocol).
    """
    if dataset_name == "I":
        dataset = ImageNet(root_path, preprocess)
        test_loader = torch.utils.data.DataLoader(
            dataset.test,
            batch_size=1,
            num_workers=8,
            shuffle=shuffle,
            worker_init_fn=_worker_init_fn,
        )
        return test_loader, dataset.classnames, dataset.template

    elif dataset_name in ["A", "V", "R", "S"]:
        preprocess = _get_ood_preprocess()
        dataset = build_dataset(f"imagenet-{dataset_name.lower()}", root_path)
        test_loader = build_data_loader(
            data_source=dataset.test,
            batch_size=1,
            is_train=False,
            tfm=preprocess,
            shuffle=shuffle,
        )

        if dataset_name in ["A", "R", "S"]:
            from .dataset_helpers import get_imagenet_subset_remap

            imagenet_classnames, subset_to_imagenet, _ = get_imagenet_subset_remap(
                dataset_name, root_path, dataset.classnames
            )
            imagenet_classnames_list = imagenet_classnames
            remapped_loader = _LabelRemapLoader(test_loader, subset_to_imagenet)
            imagenet_template = dataset.template
            return remapped_loader, imagenet_classnames_list, imagenet_template

        return test_loader, dataset.classnames, dataset.template

    elif dataset_name in [
        "caltech101",
        "dtd",
        "eurosat",
        "fgvc",
        "food101",
        "oxford_flowers",
        "oxford_pets",
        "stanford_cars",
        "sun397",
        "ucf101",
    ]:
        dataset = build_dataset(dataset_name, root_path)
        test_loader = build_data_loader(
            data_source=dataset.test,
            batch_size=1,
            is_train=False,
            tfm=preprocess,
            shuffle=shuffle,
        )
        return test_loader, dataset.classnames, dataset.template

    else:
        raise ValueError(
            f"Dataset '{dataset_name}' is not in the chosen list. "
            "Supported: I, A, V, R, S, caltech101, dtd, eurosat, fgvc, "
            "food101, oxford_flowers, oxford_pets, stanford_cars, sun397, ucf101"
        )
