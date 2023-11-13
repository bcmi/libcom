from torch import nn
import torch
import math
from typing import List

class latent_guidance_predictor(nn.Module):
    def __init__(self, output_chan, input_chan, num_encodings):
        super(latent_guidance_predictor, self).__init__()
        self.num_encodings = num_encodings
        
        self.layers = nn.Sequential(
            nn.Conv2d(input_chan, 4, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=4),
            nn.Flatten(start_dim=1, end_dim=3),
            nn.Linear(64*64*4, 512),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=512),
            nn.Linear(512, 256),         
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Linear(256, 128),     
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, 64),      
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(64, output_chan*64*64)
        )

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, x):
        # Concatenate input pixels with noise level t and positional encodings
        # pos_encoding = [torch.sin(2 * math.pi * t * (2 **-l)) for l in range(self.num_encodings)]
        # pos_encoding = torch.cat(pos_encoding, dim=-1)
        # x = torch.cat((x, t, pos_encoding), dim=-1)
        # x = x.flatten(start_dim=1, end_dim=3)
        
        return self.layers(x)
    
def resize_and_concatenate(activations: List[torch.Tensor], reference):
    assert all([isinstance(acts, torch.Tensor) for acts in activations])
    size = reference.shape[2:]
    resized_activations = []
    for acts in activations:
        acts = nn.functional.interpolate(
            acts, size=size, mode="bilinear"
        )
        # acts = acts[:1]
        # acts = acts.transpose(1,3) # b*64*64*320
        resized_activations.append(acts)

    return torch.cat(resized_activations, dim=1)

def save_tensors(module: nn.Module, features, name: str):
    """ Process and save activations in the module. """
    if type(features) in [list, tuple]:
        features = [f.detach().float() for f in features if f is not None and isinstance(f, torch.Tensor)] 
        setattr(module, name, features)
    elif isinstance(features, dict):
        features = {k: f.detach().float() for k, f in features.items()}
        setattr(module, name, features)
    else:
        setattr(module, name, features.detach().float())

def save_out_hook(self, inp, out):
    # print("hooker working")
    save_tensors(self, out, 'activations')
    return out

def predict_shadow_mask(model):
    save_hook = save_out_hook
    blocks = [0,1,2,3]
    feature_blocks = []
    for idx, block in enumerate(model.down_blocks):
        if idx in blocks:
            h=block.register_forward_hook(save_hook)
            feature_blocks.append([block,h]) 
            
    for idx, block in enumerate(model.up_blocks):
        if idx in blocks:
            h=block.register_forward_hook(save_hook)
            feature_blocks.append([block,h])
