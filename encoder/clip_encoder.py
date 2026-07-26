from .encoder import *

class CLIPSurgeryEncoder(Encoder):
    """
    CLIP Surgery Encoder that loads a CLIP Surgery model for text and image encoding.
    """

    # CS-ViT-B/16 surgery-modified attention produces locally discriminative patch
    # tokens where cosine similarity with text points in the correct direction.
    is_surgery_encoder = True

    def __init__(self, model_type='ViT-B/32', device='cpu'):
        if not hasattr(self, "initialized"):  # Only initialize once
            super().__init__(model_type, device)
            self.model, clip_preprocess = clip_surgery.load(f"CS-{model_type}", device=device)
            # Use the preprocess from clip_surgery.load() — it includes Resize to match
            # the model's input resolution (224 for ViT-B/16). This handles varying
            # image sizes (e.g. Caltech101) correctly.
            self.preprocess = clip_preprocess
            self.preprocess_for_torch = transform_for_torch(self.image_size)
            self.model.eval()
            self.initialized = True

    def encode_text(self, texts:List[str], prompt_templates=None):
        if prompt_templates is None:
            prompt_templates = ['a photo of {}']
        with torch.no_grad():
            text_features = clip_surgery.encode_text_with_prompt_ensemble(
                self.model, texts, self.device, prompt_templates=prompt_templates
            )
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def _encode_image(self, img_tensors: List, CLS_token_only=True, normalize_image_embeddings=True):
        with torch.no_grad():
            with self.timer.trace("moving_to_GPU"):
                if isinstance(img_tensors, List):
                    img_tensors = torch.stack(img_tensors)
                img_tensors = img_tensors.to(self.device)

            with self.timer.trace("image_encoding"):
                img_features = self.model.encode_image(img_tensors)
                if img_features.dim() == 3:
                    if CLS_token_only:
                        img_features = img_features[:, 0, :]  # CLS only -> [B, D]
                    else:
                        img_features = img_features  # CLS + patches -> [B, 197, D]
                if normalize_image_embeddings:
                    img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        return img_features


class CLIPEncoder(CLIPSurgeryEncoder):
    """
    Standard CLIP Encoder (not CLIP Surgery).

    Loads a vanilla OpenAI CLIP model via the CLIP Surgery library's loader
    (which supports both regular CLIP and CLIP Surgery checkpoints).
    """

    # Standard CLIP bidirectional attention causes patch tokens to absorb global
    # context, making cosine similarity with text inverted relative to CS-ViT.
    # Downstream scoring functions check this flag and negate accordingly.
    is_surgery_encoder = False

    def __init__(self, model_type='ViT-B/16', device='cpu'):
        if not hasattr(self, "initialized"):
            # Skip CLIPSurgeryEncoder.__init__ to avoid "CS-" prefix loading;
            # call Encoder.__init__ directly for transforms/setup.
            super(CLIPSurgeryEncoder, self).__init__(model_type, device)
            self.model, clip_preprocess = clip_surgery.load(model_type, device=device)
            self.preprocess = clip_preprocess
            self.preprocess_for_torch = transform_for_torch(self.image_size)
            self.model.eval()
            self.initialized = True


class OpenClipEncoder(Encoder):
    def __init__(self, model_type='ViT-B-32', device='cpu'):
        if not hasattr(self, "initialized"):  # Only initialize once
            super().__init__(model_type, device)
            self.model, _, _ = open_clip.create_model_and_transforms(model_type, pretrained='openai')
            self.tokenizer = open_clip.get_tokenizer(model_type)
            self.preprocess = transform()
            self.preprocess_for_torch = transform_for_torch(self.image_size)
            self.model.eval()
            self.model.to(self.device)
            self.initialized = True


class DetailClipEncoder(Encoder):
    def __init__(self, model_type='ViT-B/16', device='cpu'):
        import detail_clip as clip
        from detail_clip import tokenize
        if not hasattr(self, "initialized"):  # Only initialize once
            super().__init__(model_type, device)
            # checkpoint_path is read from DETAILCLIP_CHECKPOINT env var inside clip.load()
            self.model, preprocess, preprocess_for_torch = clip.load(model_type, device)
            self.tokenizer = tokenize
            # Override base class transforms — DetailCLIP expects 224×224, not 448
            self.image_size = 224
            self.preprocess = preprocess
            self.preprocess_for_torch = preprocess_for_torch

            self.model.eval()
            self.initialized = True

    def encode_text(self, texts: List[str]):
        texts = ['a photo of {}'.format(t) for t in texts]
        with torch.no_grad():
            texts = self.tokenizer(texts).to(self.device)
            if len(texts.shape) == 1:
                texts = texts.unsqueeze(0)
            text_features = self.model.encode_text(texts)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features
    
    def _encode_image_with_context_concat(self, img_tensors: List):
        with torch.no_grad():
            with self.timer.trace("moving_to_GPU"):
                if isinstance(img_tensors, List):
                    img_tensors = torch.stack(img_tensors)
                img_tensors = img_tensors.to(self.device)

            with self.timer.trace("image_encoding"):
                img_features, _, _, _ = self.model.visual(img_tensors)
                img_features = img_features @ self.model.image_projection
                img_features = img_features / img_features.norm(dim=1, keepdim=True)

                cls_emb = img_features[:,:1]
                patch_emb = img_features[:,1:]

                cls_emb_expanded = cls_emb.expand(-1, patch_emb.size(1), -1)  # (batch_size, num_patches, embedding_dim)
                patch_emb_w_context = torch.cat([cls_emb_expanded, patch_emb], dim=-1)     # (batch_size, num_patches, embedding_dim*2)

        return patch_emb_w_context

    def _encode_image(self, img_tensors: List, CLS_token_only=True, normalize_image_embeddings=True):
        with torch.no_grad():
            with self.timer.trace("moving_to_GPU"):
                if isinstance(img_tensors, List):
                    img_tensors = torch.stack(img_tensors)
                img_tensors = img_tensors.to(self.device)

            with self.timer.trace("image_encoding"):
                img_features, _, _, _ = self.model.visual_ema(img_tensors)
                img_features = img_features @ self.model.image_projection_e

                if CLS_token_only:
                    img_features = img_features[:, :1].squeeze(axis=1)  # [B, D]
                else:
                    img_features = img_features  # CLS + patches -> [B, 197, D]

                if normalize_image_embeddings:
                    img_features = img_features / img_features.norm(dim=-1, keepdim=True)
       
        return img_features

# https://github.com/UCSC-VLAA/CLIPS
class CLIPSEncoder(Encoder):
    def __init__(self, model_type='ViT-B-16', device='cpu'):
        if not hasattr(self, "initialized"):  # Only initialize once
            model_type = 'hf-hub:UCSC-VLAA/ViT-L-14-CLIPS-224-Recap-DataComp-1B'
            super().__init__(model_type, device)
            self.model, self.preprocess = open_clip.create_model_from_pretrained(model_type)
            self.tokenizer = open_clip.get_tokenizer(model_type)
            self.image_size = self.preprocess.transforms[0].size[0]
            resize = self.preprocess.transforms[0]
            normalize = self.preprocess.transforms[-1]
            self.preprocess_for_torch = Compose([resize, normalize])
            self.model.eval()
            self.model.to(self.device)
            self.initialized = True
