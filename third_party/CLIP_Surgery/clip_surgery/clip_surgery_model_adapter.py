from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
from torch import nn

import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from .clip_surgery_model import CLIPSurgery

# Sahar's CLIP Adapter
class Projection(nn.Module):
    """ CLIP Adapter Module """
    def __init__(self, d_in: int, d_out: int, p: float = 0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds
    
class ProjectionHuge(nn.Module):
    """ Deep CLIP Adapter Module """
    def __init__(self, d_in: int, d_out: int, p: float = 0.5, reduction: int = 4) -> None:
        super().__init__()
        self.down_projec = nn.Sequential(
            nn.Linear(d_in, d_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_in // reduction, d_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_in // reduction, d_in // reduction, bias=False))
        self.up_projec = nn.Sequential(
            nn.Linear(d_in // reduction, d_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_in // reduction, d_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_in // reduction, d_out, bias=False))
        
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.drop(self.down_projec(x))
        embed2 = self.drop(self.up_projec(embed1))
        embeds = self.layer_norm(embed2 + x)
        return embeds

# https://github.com/gaopengcuhk/CLIP-Adapter/blob/main/clip_adapter.py
class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class CLIPSurgeryAdapter(CLIPSurgery):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size, context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)

        self.ratio = 0.2
        # self.vadapter = Adapter(embed_dim, 4)
        # self.tadapter = Adapter(embed_dim, 4)
        self.vadapter = Projection(embed_dim, embed_dim)
        self.tadapter = Projection(embed_dim, embed_dim)


        # set attribute of adapter layers so we don't freeze them
        setattr(self.vadapter, "is_cloned", True)
        setattr(self.tadapter, "is_cloned", True)
        for name, m in self.vadapter.named_modules():
            setattr(m, "is_cloned", True)
        for name, m in self.tadapter.named_modules():
            setattr(m, "is_cloned", True)


    def _freeze_all(self):
        """Freeze the model, except for the adapter."""
        for name, m in self.named_modules():
            if isinstance(m, _BatchNorm):
                m.eval()

            # # if it's a projection layer, don't freeze it
            # if name.endswith("projection"):
            #     continue

            # if isinstance(m, nn.Conv2d):
            #     is_cloned = getattr(m, "is_cloned", False)
            #     if is_cloned:
            #         for param in m.parameters(): # for some reason things get reset to all False
            #             param.requires_grad = True
            #         continue

            is_cloned = getattr(m, "is_cloned", False)
            if not is_cloned:
                for param in m.parameters():
                    param.requires_grad = False
            else:
                for param in m.parameters():
                    param.requires_grad = True

        return

    def train(self, mode=True):
        """Convert the model into training mode while keep the normalization
        layer freezed."""
        super().train(mode)
        # if self.freeze_all:
        self._freeze_all()  # freeze everything except for clip adapter
        return self
    
    def encode_image(self, image):
        image_features = self.visual(image.type(self.dtype))
        x = self.vadapter(image_features)  # apply adapter layers after vision encoder
        image_features = self.ratio * x + (1 - self.ratio) * image_features
        return image_features

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        x_adapter = self.tadapter(x)  # apply adapter layers after text encoder
        x = self.ratio * x_adapter + (1 - self.ratio) * x

        return x





    # def _freeze_all(self):
    #     """Freeze the model."""
    #     for m in self.modules():
    #         if isinstance(m, _BatchNorm):
    #             m.eval()
    #         for param in m.parameters():
    #             param.requires_grad = False


    # def train(self, mode=True):
    #     # ensure that the model is frozen
    #     super(CLIPSurgeryAdapter, self).train(mode)
    #     self._freeze_all()


    

    #     self.adapter = nn.Linear(embed_dim, embed_dim)
    #     self.adapter_ln = LayerNorm(embed_dim)

    # def forward(self, image, text):
    #     image_features = self.encode_image(image)
    #     text_features = self.encode_text(text)

    #     # normalized features
    #     image_features = image_features / image_features.norm(dim=1, keepdim=True)
    #     text_features = text_features / text_features.norm(dim=1, keepdim=True)

    #     # apply adapter
    #     image_features = self.adapter_ln(self.adapter(image_features))
    #     text_features = self.adapter_ln(self.adapter(text_features))

    #     # cosine similarity as logits
    #     logit_scale = self.logit_scale.exp()
    #     logits_per_image = logit_scale * image_features @ text_features.t()
    #     logits_per_text = logits_per_image.t()

    #     # shape = [global_batch_size, global_batch_size]
    #     return logits_per_image, logits_per_text