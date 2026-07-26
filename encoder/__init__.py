from .encoder import *
from .clip_encoder import *
# from .dino_encoder import *
# from .open_vision_encoder import *
# from .ad_encoder import *

ENCODER_MAP = {
    'clip_surgery': CLIPSurgeryEncoder,
    'clip': CLIPEncoder,
    # 'open-clip': OpenClipEncoder,w
    # 'bridge_tower': BridgeTowerEncoder,
    'detail-clip': DetailClipEncoder,
    # 'open-vision': OpenVisionEncoder,
    # 'clips': CLIPSEncoder,
}

def get_encoder_type(encoder: Encoder):
    encoder_type = ''
    for k in ENCODER_MAP.keys():
        if ENCODER_MAP[k] == type(encoder):
            encoder_type = k
            break
    return encoder_type


def create_encoder_instance(encoder_type: str, **kwargs):
    cls = ENCODER_MAP.get(encoder_type, None)
    if cls is None:
        return None
    encoder = cls(**kwargs)
    return encoder

