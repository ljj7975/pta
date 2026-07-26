import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import *

from .encoder import Encoder
from third_party.ad_pretrain.models.dino import DinoModel
from third_party.ad_pretrain.models.clip import ClipModel
from third_party.ad_pretrain.models.imagebind import ImageBindModel
from third_party.ad_pretrain.models.projector import MultiScaleAttentionProjector


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def dino_transform():
    return Compose([
        ToTensor(),
        Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def dino_transform_for_torch(n_px):
    return Compose([
        Resize(n_px, interpolation=InterpolationMode.BILINEAR),
        Lambda(lambda img: TF.pad(
            img,
            padding=[
                (max(TF.get_image_size(img)) - TF.get_image_size(img)[0]) // 2,
                (max(TF.get_image_size(img)) - TF.get_image_size(img)[1]) // 2,
                (max(TF.get_image_size(img)) - TF.get_image_size(img)[0] + 1) // 2,
                (max(TF.get_image_size(img)) - TF.get_image_size(img)[1] + 1) // 2,
            ],
            fill=0,
            padding_mode="constant",
        )),
        Resize((n_px, n_px), interpolation=InterpolationMode.BILINEAR),
        Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ADPretrainedEncoder(Encoder):
    """
    AD-pretrained DINO encoder.

    Pipeline:
        image
        -> DINO backbone intermediate feature maps
        -> MultiScaleAttentionProjector
        -> merge projected multi-scale features
        -> flatten to patch tokens
    """

    def __init__(
        self,
        model_type: str = "ad_pretrained_dinov2-base",
        device: str = "cuda:0",
        backbone_name: str = "dinov2-base",
        pretrained_weights: str = "/data2/public_data/pretrained_models/ad_pretrain/dinov2/checkpoints_pro_angle.pth",
        image_size: int = 224,
        merge_mode: str = "mean",
    ):
        super().__init__(model_type=model_type, device=device)

        self.model_type = model_type
        self.device = device
        self.image_size = image_size
        self.merge_mode = merge_mode

        # Override preprocessing from base Encoder:
        # AD-pretrained DINO path expects ImageNet normalization.
        self.preprocess = dino_transform()
        self.preprocess_for_torch = dino_transform_for_torch(self.image_size)
        self.preprocess_mask_for_torch = Compose([
            Lambda(lambda t: t.unsqueeze(0)),
            *self.preprocess_for_torch.transforms[:-1],
            Lambda(lambda t: t[0]),
        ])

        if 'dino' in backbone_name:
            self.backbone = DinoModel(name=backbone_name, device=device).to(device)
        elif 'clip' in backbone_name:
            self.backbone = ClipModel(name=backbone_name, device=device)
        elif 'imagebind' in backbone_name:
            self.backbone = ImageBindModel(name=backbone_name, device=device)
        self.backbone.eval()

        self.projector = MultiScaleAttentionProjector(
            channels=self.backbone.feature_dimensions,
            device=device,
        ).to(device)
        self.projector.eval()

        checkpoint = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
        if "projectors" not in checkpoint:
            raise KeyError(
                f"'projectors' not found in checkpoint: {pretrained_weights}. "
                f"Available keys: {list(checkpoint.keys())}"
            )
        self.projector.load_state_dict(checkpoint["projectors"], strict=True)
        self.projector.eval()

    @property
    def embedding_dim(self):
        return self.backbone.feature_dimensions[-1]

    def _merge_projected_features(self, projected_features, cls_tokens):
        """
        projected_features: tuple/list of 4 tensors, each [B, C, H, W]

        Returns:
            merged tensor [B, C, H, W]
        """
        if self.merge_mode == "mean":
            return torch.stack(cls_tokens, dim=0).mean(dim=0), torch.stack(projected_features, dim=0).mean(dim=0)

        elif self.merge_mode == 'last':
            return cls_tokens[-1], projected_features[-1]
        
        elif self.merge_mode == "concat":
            # TODO
            pass

        else:
            raise ValueError(f"Unsupported merge_mode: {self.merge_mode}")

    def _feature_map_to_tokens(self, feat_map, cls_token, normalize_image_embeddings=True):
        """
        feat_map: [B, C, H, W]

        Returns:
            img_emb: [B, 1 + N, D]
        """
        b, c, h, w = feat_map.shape

        # [B, C, H, W] -> [B, N, C]
        patch_tokens = feat_map.flatten(2).transpose(1, 2)

        img_emb = torch.cat([cls_token, patch_tokens], dim=1)  # [B, 1+N, D]

        if normalize_image_embeddings:
            img_emb = F.normalize(img_emb, dim=-1)

        return img_emb

    @torch.no_grad()
    def _encode_image(self, img_tensors, CLS_token_only=True, normalize_image_embeddings=True):
        if isinstance(img_tensors, list):
            img_tensors = torch.stack(img_tensors)

        img_tensors = img_tensors.to(self.device)

        features, cls_tokens = self.backbone.encode_image_from_tensors(
            img_tensors,
        )

        if not isinstance(features, (list, tuple)) or len(features) != 4:
            raise ValueError(
                f"Expected 4 feature maps from DinoModel, got type={type(features)}"
            )

        p1, p2, p3, p4 = self.projector(*features, keep_shape=True)

        cls_merged, patch_merged = self._merge_projected_features((p1, p2, p3, p4), cls_tokens)

        img_emb = self._feature_map_to_tokens(
            patch_merged,
            cls_merged,
            normalize_image_embeddings=normalize_image_embeddings,
        )

        if CLS_token_only:
            return img_emb[:, 0]  # [B, D]

        return img_emb  # [B, 1+N, D]