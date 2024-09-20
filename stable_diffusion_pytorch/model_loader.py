import torch
from . import Tokenizer, CLIP, Encoder, Decoder, Diffusion
from . import util
import warnings

def make_compatible(state_dict):
    keys = list(state_dict.keys())
    changed = False
    for key in keys:
        if "causal_attention_mask" in key:
            del state_dict[key]
            changed = True
        elif "_proj_weight" in key:
            new_key = key.replace('_proj_weight', '_proj.weight')
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
            changed = True
        elif "_proj_bias" in key:
            new_key = key.replace('_proj_bias', '_proj.bias')
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
            changed = True

    if changed:
        warnings.warn(("Given checkpoint data were modified dynamically by make_compatible"
                       " function on model_loader.py. Maybe this happened because you're"
                       " running newer codes with older checkpoint files. This behavior"
                       " (modify old checkpoints and notify rather than throw an error)"
                       " will be removed soon, so please download latest checkpoints file."))

    return state_dict

def load_clip(device, dtype=torch.float32):
    state_dict = torch.load(util.get_file_path('stable-diffusion-pt/clip.pt'))
    state_dict = make_compatible(state_dict)

    clip = CLIP().to(device, dtype=dtype)
    clip.load_state_dict(state_dict)
    return clip

def load_encoder(device, dtype=torch.float32):
    state_dict = torch.load(util.get_file_path('stable-diffusion-pt/encoder.pt'))
    state_dict = make_compatible(state_dict)

    encoder = Encoder().to(device, dtype=dtype)
    encoder.load_state_dict(state_dict)
    return encoder

def load_decoder(device, dtype=torch.float32):
    state_dict = torch.load(util.get_file_path('stable-diffusion-pt/decoder.pt'))
    state_dict = make_compatible(state_dict)

    decoder = Decoder().to(device, dtype=dtype)
    decoder.load_state_dict(state_dict)
    return decoder

def load_diffusion(device, dtype=torch.float32):
    state_dict = torch.load(util.get_file_path('stable-diffusion-pt/diffusion.pt'))
    state_dict = make_compatible(state_dict)

    diffusion = Diffusion().to(device, dtype=dtype)
    diffusion.load_state_dict(state_dict)
    return diffusion

def preload_models(device, dtype=torch.float32):
    return {
        'clip': load_clip(device, dtype),
        'encoder': load_encoder(device, dtype),
        'decoder': load_decoder(device, dtype),
        'diffusion': load_diffusion(device, dtype),
    }
