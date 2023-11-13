import os
import torch

from omegaconf import OmegaConf
from libcom.shadow_generation.source.ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    return state_dict


def create_model(config_path):
    config = OmegaConf.load(config_path) if isinstance(config_path, str) else config_path
    model  = instantiate_from_config(config.model).cpu()
    return model
