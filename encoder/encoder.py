import sys
import torch
import torch.nn as nn
import warnings
from typing import List, Union
from third_party.CLIP_Surgery.clip_surgery import clip as clip_surgery
from torchvision.transforms import *
from utils.timer import Timer

import torch.nn.functional as F
import torchvision.transforms.functional as TF
from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning

NORMALIZATION_MEAN = (0.48145466, 0.4578275, 0.40821073)
NORMALIZATION_STD = (0.26862954, 0.26130258, 0.27577711)


def convert_image_to_rgb(image):
    return image.convert("RGB")


def transform():
    return Compose([
        ToTensor(),
        Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
    ])


def transform_for_torch(n_px):
    return Compose([
        Resize(n_px, interpolation=InterpolationMode.BILINEAR),
        Lambda(lambda img: TF.pad(
            img,
            padding=[
                (max(TF.get_image_size(img)) - TF.get_image_size(img)[0]) // 2,
                (max(TF.get_image_size(img)) - TF.get_image_size(img)[1]) // 2,
                (max(TF.get_image_size(img)) - TF.get_image_size(img)[0] + 1) // 2,
                (max(TF.get_image_size(img)) - TF.get_image_size(img)[1] + 1) // 2
            ],
            fill=0,
            padding_mode="constant"
        )),
        Resize((n_px, n_px), interpolation=InterpolationMode.BILINEAR),
        Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
    ])


def inverse_transform(patch_size):
    mean = torch.as_tensor(NORMALIZATION_MEAN)
    std = torch.as_tensor(NORMALIZATION_STD)
    std_inv = 1 / (std + 1e-7)
    mean_inv = -mean * std_inv
    return Compose([Normalize(mean_inv, std_inv),
                    Resize(patch_size, interpolation=InterpolationMode.BICUBIC)])

class Encoder(nn.Module):
    """
    Singleton class for encoding models. Ensures that only one instance is created.
    """
    _instances = {}

    def __init__(self, model_type:str, device:str):
        super().__init__()

        self.timer = Timer()

        self.model_type = model_type
        self.device = device
        # self.image_size = 224
        self.image_size = 448
        self.preprocess = transform()
        self.preprocess_for_torch = transform_for_torch(self.image_size)
        self.preprocess_mask_for_torch = Compose([
                transforms.Lambda(lambda t: t.unsqueeze(0)),
                *self.preprocess_for_torch.transforms[:-1],
                transforms.Lambda(lambda t: t[0])
            ])

    def __new__(cls, *args, **kwargs):
        inst_id = f"{cls.__name__}"
        if "model_type" in kwargs:
            model_type = kwargs["model_type"]
            inst_id = f"{inst_id}-{model_type}"
        else:
            warnings.warn(f"model_type is not specified for instantiating {inst_id} singleton")
        if inst_id not in cls._instances:
            cls._instances[inst_id] = super(Encoder, cls).__new__(cls)
        return cls._instances[inst_id]

    def encode_text(self, texts: List[str]):
        texts = ['a photo of {}'.format(t) for t in texts]
        with torch.no_grad():
            texts = self.tokenizer(texts).to(self.device)
            text_features = self.model.encode_text(texts, normalize=True)
        return text_features

    def _wrap_encode_image(self, x: torch.Tensor):
        model = self.model.visual
        x = model.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + model.positional_embedding.to(x.dtype)
        x = model.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = model.ln_post(x) # return both cls token and image tokens

        if model.proj is not None:
            x = x @ model.proj

        return x
    
    @property
    def visual(self):
        return self.model.visual

    def get_patch_embeddings(self, image, exclude_pos=False):
        """Extract per-patch embeddings (strips CLS token).

        Returns [P, D] where P = num patches (e.g. 196 for ViT-B/16).
        """
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            features = self.encode_image(image, CLS_token_only=False, preprocess=False)
            patches = features[:, 1:]
            return patches.squeeze(0)

    def _encode_image(self, img_tensors: List, CLS_token_only=True, normalize_image_embeddings=True):
        with torch.no_grad():
            with self.timer.trace("moving_to_GPU"):
                if isinstance(img_tensors, List):
                    img_tensors = torch.stack(img_tensors)
                img_tensors = img_tensors.to(self.device)

            with self.timer.trace("image_encoding"):
                img_features = self._wrap_encode_image(img_tensors)

                if CLS_token_only:
                    img_features = img_features[:, :1].squeeze(axis=1)  # [B, D]
                else:
                    img_features = img_features  # [B, 1+P, D] — CLS + patches

                if normalize_image_embeddings:
                    img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        return img_features
    
    def _encode_image_with_context_concat(self, img_tensors: List):
        raise NotImplementedError
    
    def preprocess_image(self, imgs:List):
        with torch.no_grad():
            if torch.is_tensor(imgs) or torch.is_tensor(imgs[0]):
                for img in imgs:
                    assert img.max() <= 255 and img.min() >= 0
                img_tensors = [self.preprocess_for_torch(img) for img in imgs]
            else:
                img_tensors = [self.preprocess(img) for img in imgs]
        return img_tensors
    
    def preprocess_mask(self, masks:List):
        with torch.no_grad():
            if torch.is_tensor(masks) or torch.is_tensor(masks[0]):
                if hasattr(self, "preprocess_mask_for_torch"):
                    img_tensors = [self.preprocess_mask_for_torch(mask) for mask in masks]
                else:
                    raise ValueError("Missing preprocess_mask_for_torch operation")
            else:
                if hasattr(self, "preprocess_mask"):
                    img_tensors = [self.preprocess_mask(mask) for mask in masks]
                else:
                    raise ValueError("Missing preprocess_mask operation")
        return img_tensors

    def encode_image(self, imgs: List, CLS_token_only=True, preprocess=False, context_mode=None, normalize_image_embeddings=True):
        with self.timer.trace("preprocessing"):
            if preprocess:
                img_tensors = self.preprocess_image(imgs)
            else:
                img_tensors = imgs
                
        if context_mode is None:
            outs = self._encode_image(img_tensors, CLS_token_only, normalize_image_embeddings=normalize_image_embeddings)
        elif context_mode == 'concat':
            outs = self._encode_image_with_context_concat(img_tensors)
        else:
            raise ValueError(f"Invalid context mode: {context_mode}")
        return outs