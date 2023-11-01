import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.contiguous().view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adain_fg(comp_feat, mask):
    #assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = comp_feat.size()
    #style_mean, style_std = calc_mean_std(style_feat)  # the style features
    downsample_mask_style = 1 - mask
    style_mean, style_std = get_foreground_mean_std(comp_feat, downsample_mask_style)  # the style features
    fore_mean, fore_std = get_foreground_mean_std(comp_feat, mask)  # the foreground features

    normalized_feat = (comp_feat - fore_mean.expand(size)) / fore_std.expand(size)
    return (normalized_feat * style_std.expand(size) + style_mean.expand(size)) * mask + (comp_feat * (1 - mask))


decoder_cat = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(), # relu1-1
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 4
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(), # relu2-1
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 17
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 24
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Conv2d(65, 1, (1, 1),padding=0,stride=1), ##matting layer
    nn.ReflectionPad2d((1, 1, 1, 1)), # 29
    nn.Conv2d(64, 3, (3, 3)),
)


vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)), 
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)



def get_foreground_mean_std(features, mask, eps=1e-5):
    region = features * mask 
    sum = torch.sum(region, dim=[2, 3])     # (B, C)
    num = torch.sum(mask, dim=[2, 3])       # (B, C)
    mu = sum / (num + eps)
    mean = mu[:, :, None, None]
    var = torch.sum((region + (1 - mask)*mean - mean) ** 2, dim=[2, 3]) / (num + eps)
    var = var[:, :, None, None]
    std = torch.sqrt(var+eps)
    return mean, std


class PHDNet(nn.Module):
    def __init__(self, vgg, decoder):
        super(PHDNet, self).__init__()
        # load the pretrained VGG encoder
        vgg_layers = list(vgg.children())
        self.enc_1 = nn.Sequential(*vgg_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*vgg_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*vgg_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*vgg_layers[18:31])  # relu3_1 -> relu4_1
       

        # define the decoder
        self.decoder = decoder
        dec_layers = list(decoder.children())
        self.dec_1 =  nn.Sequential(*dec_layers[:4]) 
        self.dec_2 =  nn.Sequential(*dec_layers[4:17]) 
        self.dec_3 =  nn.Sequential(*dec_layers[17:24]) 
        self.dec_4 =  nn.Sequential(*dec_layers[24:27]) 
        self.conv_attention = nn.Sequential(*dec_layers[27:28]) 
        self.dec_4_2 =  nn.Sequential(*dec_layers[28:]) 

        # fix the VGG encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False


    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
    
    
    def decode(self, comp, mask, enc_features):
        width = height = enc_features[-1].size(-1)
        downsample_mask = self.downsample(mask, width, height)
        t = adain_fg(enc_features[-1], downsample_mask)
        dec_feature = self.dec_1(t)

        for i in range(1,4):
            func = getattr(self, 'dec_{:d}'.format(i + 1))
            width = height = enc_features[-(i+1)].size(-1)
            downsample_mask = self.downsample(mask, width, height)
            t = adain_fg(enc_features[-(i+1)], downsample_mask)
            dec_feature = func(torch.cat([dec_feature, t], dim=1))

        
        width = height = dec_feature.size(-1)
        downsample_mask = self.downsample(mask, width, height)

        attention_map = torch.sigmoid(self.conv_attention(torch.cat([dec_feature, downsample_mask], dim=1)))
        coarse_output = self.dec_4_2(dec_feature)
        output = attention_map * coarse_output + (1.0 - attention_map) * comp
        return output


    def downsample(self, image_tensor, width, height):
        image_upsample_tensor = torch.nn.functional.interpolate(image_tensor, size=[width, height])
        image_upsample_tensor = image_upsample_tensor.clamp(0, 1)
        return image_upsample_tensor


    def forward(self, comp, mask):
        comb_feats = self.encode_with_intermediate(comp)
        final_output = self.decode(comp, mask, comb_feats)

        return final_output

