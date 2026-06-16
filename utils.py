import os
import yaml
import torch
import numpy as np
import clip
from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader, AugMixAugmenter
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []
        
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights

def get_clip_logits(images, clip_model, clip_weights):
    with torch.no_grad():
        if isinstance(images, list):
            images = torch.cat(images, dim=0).cuda()
        else:
            images = images.cuda()

        image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        clip_logits = 100. * image_features @ clip_weights

        if image_features.size(0) > 1:
            batch_entropy = softmax_entropy(clip_logits)
            selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.1)]
            output = clip_logits[selected_idx]
            image_features = image_features[selected_idx].mean(0).unsqueeze(0)
            clip_logits = output.mean(0).unsqueeze(0)

            loss = avg_entropy(output)
            prob_map = output.softmax(1).mean(0).unsqueeze(0)
            pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
        else:
            loss = softmax_entropy(clip_logits)
            prob_map = clip_logits.softmax(1)
            pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])

        return image_features, clip_logits, loss, prob_map, pred

def get_ood_preprocess():
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
    base_transform = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    aug_preprocess = AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=True)

    return aug_preprocess


def get_config_file(config_path, dataset_name):
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
        raise FileNotFoundError(f"The configuration file {config_file} was not found.")

    with open(config_file, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.SafeLoader)

    return cfg


def get_imagenet_subset_remap(dataset_name, root_path, subset_classnames):
    """
    For imagenet-r/a/s datasets, return two things needed for paper-compatible eval:
      - imagenet_classnames: full 1000-class name list (ordered by imagenet label 0..999)
      - subset_to_imagenet: LongTensor of length 200 mapping subset label -> imagenet label
        (so imagenet_logits[:, subset_to_imagenet] gives per-sample subset-class logits)
      - imagenet_to_subset: dict mapping imagenet label -> subset label (-1 if not in subset)

    This allows evaluating with 1000-class CLIP weights and then remapping predictions
    to the 200-class subset label space, matching the paper's evaluation protocol.
    """
    imagenet_classnames_dict = ImageNet.read_classnames(
        os.path.join(root_path, "imagenet", "classnames.txt")
    )
    imagenet_synsets = list(imagenet_classnames_dict.keys())
    imagenet_classnames = [imagenet_classnames_dict[s] for s in imagenet_synsets]
    imagenet_idx = {s: i for i, s in enumerate(imagenet_synsets)}

    dataset_dir_map = {
        'A': os.path.join(root_path, "imagenet-adversarial", "imagenet-a"),
        'R': os.path.join(root_path, "imagenet-rendition", "imagenet-r"),
        'S': os.path.join(root_path, "imagenet-sketch", "images"),
    }
    from datasets.utils import listdir_nohidden
    image_dir = dataset_dir_map[dataset_name]
    folders = sorted([f for f in listdir_nohidden(image_dir, sort=True) if f not in ['README.txt']])

    subset_to_imagenet = []
    for folder in folders:
        if folder in imagenet_idx:
            subset_to_imagenet.append(imagenet_idx[folder])
        else:
            subset_to_imagenet.append(-1)

    subset_to_imagenet = torch.tensor(subset_to_imagenet, dtype=torch.long)
    imagenet_to_subset = {im_lbl: sub_lbl for sub_lbl, im_lbl in enumerate(subset_to_imagenet.tolist()) if im_lbl >= 0}

    return imagenet_classnames, subset_to_imagenet, imagenet_to_subset


def build_test_data_loader(dataset_name, root_path, preprocess, shuffle=True):
    if dataset_name == 'I':
        dataset = ImageNet(root_path, preprocess)
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=1, num_workers=8, shuffle=shuffle)
        return test_loader, dataset.classnames, dataset.template

    elif dataset_name in ['A', 'V', 'R', 'S']:
        preprocess = get_ood_preprocess()
        dataset = build_dataset(f"imagenet-{dataset_name.lower()}", root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=shuffle)

        if dataset_name in ['A', 'R', 'S']:
            imagenet_classnames, subset_to_imagenet, _ = get_imagenet_subset_remap(
                dataset_name, root_path, dataset.classnames
            )
            imagenet_classnames_list = imagenet_classnames
            remapped_loader = _LabelRemapLoader(test_loader, subset_to_imagenet)
            imagenet_template = dataset.template
            return remapped_loader, imagenet_classnames_list, imagenet_template

        return test_loader, dataset.classnames, dataset.template

    elif dataset_name in ['caltech101', 'dtd', 'eurosat', 'fgvc', 'food101', 'oxford_flowers', 'oxford_pets', 'stanford_cars', 'sun397', 'ucf101']:
        dataset = build_dataset(dataset_name, root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=shuffle)
        return test_loader, dataset.classnames, dataset.template

    else:
        raise "Dataset is not from the chosen list"


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