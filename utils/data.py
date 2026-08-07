"""Data loaders, transforms, and worker utilities."""

import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader, AugMixAugmenter, Datum

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


def build_test_data_loader(dataset_name: str, root_path: str, preprocess, shuffle: bool = True, seed: int = None):
    """Build a test DataLoader for the given dataset.

    Returns ``(test_loader, classnames, template)``.

    For ImageNet-A/R/S the loader is wrapped with ``_LabelRemapLoader`` so that
    200-class subset labels are remapped to 1000-class ImageNet labels at
    iteration time (matching the paper evaluation protocol).

    ``seed`` (optional) makes the shuffled stream deterministic: the loader is
    built single-worker (``num_workers=0``) with an explicit
    ``torch.Generator().manual_seed(seed)`` — the stock 8-worker shuffled
    loader yields batches in nondeterministic completion order on this cluster
    (see ``tests/test_records_zeroshot.py:14-17``). When ``seed`` is None the
    legacy behavior (8 workers + worker_init_fn) is preserved. The ImageNet
    "I" loader keeps its own ``num_workers=8`` construction unchanged.
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
            seed=seed,
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
            seed=seed,
        )
        return test_loader, dataset.classnames, dataset.template

    else:
        raise ValueError(
            f"Dataset '{dataset_name}' is not in the chosen list. "
            "Supported: I, A, V, R, S, caltech101, dtd, eurosat, fgvc, "
            "food101, oxford_flowers, oxford_pets, stanford_cars, sun397, ucf101"
        )


def _parse_class_selection(class_names, class_file):
    """Resolve selected class names from either ``class_names`` or ``class_file``.

    ``class_file`` holds one classname per line; lines whose stripped content
    is empty or starts with ``#`` are ignored (comments). Returns a
    de-duplicated list preserving order of appearance.
    """
    if (class_names is None) == (class_file is None):
        raise ValueError("Provide exactly one of class_names or class_file")
    if class_file is not None:
        with open(class_file, "r") as f:
            parsed = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
        if not parsed:
            raise ValueError(f"No class names found in class file '{class_file}'")
        class_names = parsed
    # De-duplicate while preserving order (dicts keep insertion order in py3.7+).
    return list(dict.fromkeys(class_names))


def build_subset_test_data_loader(
    dataset_name: str,
    root_path: str,
    preprocess,
    class_names=None,
    class_file=None,
    shuffle: bool = True,
    seed: int = None,
):
    """Build a closed-set test DataLoader restricted to a subset of classes.

    Loads the full dataset via ``build_dataset`` (same as
    ``build_test_data_loader``), keeps only samples whose label belongs to the
    selected classes, and remaps their labels to ``0..K-1`` in class-list
    order. Selection is either ``class_names`` (list of str) or ``class_file``
    (path with one classname per line; ``#`` lines are comments); every name
    is validated against ``dataset.classnames``.

    Returns ``(subset_loader, subset_classnames, template)`` where
    ``subset_classnames`` preserves the input class-list order, so the i-th
    entry corresponds to remapped label ``i``.
    """
    class_names = _parse_class_selection(class_names, class_file)
    dataset = build_dataset(dataset_name, root_path)

    full_classnames = list(dataset.classnames)
    for name in class_names:
        if name not in full_classnames:
            raise ValueError(
                f"Unknown class name '{name}' for dataset '{dataset_name}'. "
                f"Valid classes: {full_classnames}"
            )

    # Original dataset label -> remapped label (0..K-1) in class-list order.
    label_to_new = {
        full_classnames.index(name): new_label
        for new_label, name in enumerate(class_names)
    }

    subset_items = [
        Datum(
            impath=item.impath,
            label=label_to_new[item.label],
            domain=item.domain,
            classname=item.classname,
        )
        for item in dataset.test
        if item.label in label_to_new
    ]

    subset_loader = build_data_loader(
        data_source=subset_items,
        batch_size=1,
        is_train=False,
        tfm=preprocess,
        shuffle=shuffle,
        seed=seed,
    )
    return subset_loader, list(class_names), dataset.template
