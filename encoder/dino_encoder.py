from .encoder import *
from dinov3.models.vision_transformer import vit_small

class DINOEncoder(Encoder):
    def __init__(self, model_type='', device='cpu'):
        if not hasattr(self, "initialized"):  # Only initialize once
            super().__init__(model_type, device)
            self._load_model(device=device)
            self.preprocess = transform()
            self.initialized = True

    def _load_model(self, model_type='', device='cpu'):
        raise NotImplementedError

    def _get_embeddings(self, x):
        raise NotImplementedError

    def _encode_image(self, img_tensors:List, CLS_token_only=True, normalize_image_embeddings=True):
        with torch.no_grad():
            if isinstance(img_tensors, List):
                img_tensors = torch.stack(img_tensors)
            img_tensors = img_tensors.to(self.device)
            img_features = self._get_embeddings(img_tensors, CLS_token_only)
            if normalize_image_embeddings:
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        return img_features


class DINOv1Encoder(DINOEncoder):
    def _load_model(self, model_type='dino_vitb16', device='cpu'):
        # Avoid name conflicts
        if sys.modules.get('utils') is not None:
            m = sys.modules.pop('utils')
            self.model = torch.hub.load('facebookresearch/dino:main', model_type).to(device)
            sys.modules['utils'] = m
        else:
            self.model = torch.hub.load('facebookresearch/dino:main', model_type).to(device)
        self.model.eval()
        self.model_type = model_type

    def _get_embeddings(self, x, CLS_token_only=True):
        assert CLS_token_only, 'DINOv1 supports only image level embedding'
        return self.model(x) # [batch, D]


class DINOv2Encoder(DINOEncoder):
    def _load_model(self, model_type='dinov2_vitb14', device='cpu'):
        self.model = torch.hub.load('facebookresearch/dinov2', model_type).to(device)
        self.model.eval()
        self.model_type = model_type

    def _get_embeddings(self, x, CLS_token_only=True):
        raw_outputs = self.model.forward_features(x)
        img_features = raw_outputs["x_prenorm"] 
        return img_features
    

class DINOv3Encoder(DINOEncoder):
    def _load_model(self, model_type='dinov3_vits16plus_pretrain_lvd1689m-4057cbaa', device='cpu'):
        self.model = vit_small()
        state_dict = torch.load(f"./third_party/dinov3/checkpoint/{model_type}.pth", map_location=device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(device)
        self.model.eval()
        
        self.model_type = model_type

    def _get_embeddings(self, x, CLS_token_only=True):
        raw_outputs = self.model.forward_features(x)
        img_features = raw_outputs["x_prenorm"] 
        return img_features