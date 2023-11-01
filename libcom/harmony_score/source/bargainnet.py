import torch
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import torch.nn.functional as F
from torch import nn
import functools
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import math

class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        self.return_mask = True

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)


        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output

class StyleEncoder(nn.Module):
    def __init__(self, style_dim, norm_layer=nn.BatchNorm2d):
        super(StyleEncoder, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        ndf=64
        n_layers=6
        kw = 3
        padw = 0
        self.conv1f = PartialConv2d(3, ndf, kernel_size=kw, stride=2, padding=padw)
        self.relu1 = nn.ReLU(True)
        nf_mult = 1
        nf_mult_prev = 1

        n = 1
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv2f = PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)
        self.norm2f = norm_layer(ndf * nf_mult)
        self.relu2 = nn.ReLU(True)

        n = 2
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv3f = PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)
        self.norm3f = norm_layer(ndf * nf_mult)
        self.relu3 = nn.ReLU(True)

        n = 3
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv4f = PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)
        self.norm4f = norm_layer(ndf * nf_mult)
        self.relu4 = nn.ReLU(True)

        n = 4
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv5f = PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.convs = nn.Conv2d(ndf * nf_mult, style_dim, kernel_size=1, stride=1)

    def forward(self, input, mask):
        """Standard forward."""
        xb = input
        mb = mask

        xb, mb = self.conv1f(xb, mb)
        xb = self.relu1(xb)
        xb, mb = self.conv2f(xb, mb)
        xb = self.norm2f(xb)
        xb = self.relu2(xb)
        xb, mb = self.conv3f(xb, mb)
        xb = self.norm3f(xb)
        xb = self.relu3(xb)
        xb, mb = self.conv4f(xb, mb)
        xb = self.norm4f(xb)
        xb = self.relu4(xb)
        xb, mb = self.conv5f(xb, mb)
        xb = self.avg_pooling(xb)
        s = self.convs(xb)
        s = torch.squeeze(s)
        return s

class InharmonyLevelPredictor:
    def __init__(self, device):
        self.device = device
        self.model  = self.build_inharmony_predictor()
        image_size  = 256
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def build_inharmony_predictor(self):
        model = StyleEncoder(style_dim=16)
        weight = os.path.join(os.path.dirname(__file__), 'net_E.pth')
        assert os.path.exists(weight), weight
        print('Build InharmonyLevel Predictor')
        model.load_state_dict(torch.load(weight, map_location='cpu'))
        model = model.eval().to(self.device)
        return model

    def data_preprocess(self, image, mask):
        if mask.max() > 1.:
            fg_mask = mask.astype(np.float32) / 255.
        else:
            fg_mask = mask.astype(np.float32)
        bg_mask = 1 - fg_mask
        fg_mask = self.mask_transform(Image.fromarray(fg_mask))
        fg_mask = fg_mask.unsqueeze(0).to(self.device)
        bg_mask = self.mask_transform(Image.fromarray(bg_mask))
        bg_mask = bg_mask.unsqueeze(0).to(self.device)

        image   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image   = self.transform(Image.fromarray(image))
        image   = image.unsqueeze(0).to(self.device)
        return image, bg_mask, fg_mask

    def Normalized_Euclidean_distance(self, vec1, vec2):
        vec1 = vec1.cpu().numpy()
        vec2 = vec2.cpu().numpy()
        norm_vec1 = vec1 / np.sqrt(np.maximum(np.sum(vec1**2), 1e-12))
        norm_vec2 = vec2 / np.sqrt(np.maximum(np.sum(vec2**2), 1e-12))
        cos_sim   = (norm_vec1 * norm_vec2).sum()
        angle_sim = np.arccos(cos_sim) / np.pi
        dist      = np.sqrt(1 - angle_sim)
        # print('norm_vec1**2', (norm_vec1**2).sum())
        # print('norm_vec2**2', (norm_vec2**2).sum())
        # print('v1 * c2', torch.sum(norm_vec1 * norm_vec2))
        # print('dist', torch.sqrt(2 - torch.sum(2 * norm_vec1 * norm_vec2)))
        return dist

    def Euclidean_distance(self, vec1, vec2):
        vec1 = vec1.cpu().numpy()
        vec2 = vec2.cpu().numpy()
        dist = np.sqrt(np.sum((vec1 - vec2)**2))
        return dist

    def __call__(self, image, mask):
        with torch.no_grad():
            im, bg_mask, fg_mask = self.data_preprocess(image, mask)
            bg_sty_vector = self.model(im, bg_mask)
            fg_sty_vector = self.model(im, fg_mask)
        # norm_dist = self.Normalized_Euclidean_distance(bg_sty_vector, fg_sty_vector)
        # norm_dist = round(norm_dist, 2)
        eucl_dist = self.Euclidean_distance(bg_sty_vector, fg_sty_vector)
        # convert distance to harmony level which lies in 0 and 1
        harm_level = math.exp(-0.04212 * eucl_dist)
        harm_level = round(harm_level, 2)
        return harm_level


if __name__ == "__main__":
    device = torch.device("cuda:0")
    model = InharmonyLevelPredictor(device)
    img = np.random.randint(0, 255, (256, 256, 3)).astype(np.uint8)
    mask = np.random.randint(0, 255, (256, 256)).astype(np.uint8)
    local_pre = model(img, mask)
    print(local_pre)