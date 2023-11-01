import torch
from . import networks
from torch import nn
import os

class PHDNet(nn.Module):
    def __init__(self, device):
        super(PHDNet, self).__init__()
        # define networks
        self.device = device
        self.netvgg = networks.vgg
        self.netvgg = nn.Sequential(*list(self.netvgg.children())[:31])
        self.netDecoder = networks.decoder_cat
        self.netG = networks.PHDNet(self.netvgg, self.netDecoder)


    def forward(self, comp, mask):
        """Employ generator to generate the output"""
        if len(comp.shape) == 3:
            comp = comp.unsqueeze(0)
            mask = mask.unsqueeze(0)

        self.comp = comp.to(self.device)
        self.mask = mask.to(self.device)
        
        output = self.netG(self.comp, self.mask)
        return output


    def load_networks(self, load_path):
        """Load all the networks from the disk.

        Parameters:
            load_path (str) -- path to load parameters
        """
        if os.path.exists(load_path):
            net = getattr(self, 'netG')
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            # print('loading the model from %s' % load_path)
            state_dict = torch.load(load_path, map_location=self.device)
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata

            net.load_state_dict(state_dict, strict=True)
        else:
            print('failed loading model from {}'.format(load_path))