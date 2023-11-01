import os,sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from torch import nn

from oblect_place_config import opt
from resnet_4ch import resnet

class ObjectPlaceNet(nn.Module):
    def __init__(self, backbone_pretrained=True):
        super(ObjectPlaceNet, self).__init__()
        
        ## Backbone, only resnet
        resnet_layers = int(opt.backbone.split('resnet')[-1])
        backbone = resnet(resnet_layers,
                          backbone_pretrained,
                          os.path.join(opt.pretrained_model_path, opt.backbone+'.pth'))

        # drop pool layer and fc layer
        features = list(backbone.children())[:-2]
        backbone = nn.Sequential(*features)
        self.backbone = backbone

        ## global predict
        self.global_feature_dim = 512 if opt.backbone in ['resnet18', 'resnet34'] else 2048
   
        self.avgpool3x3 = nn.AdaptiveAvgPool2d(3)
        self.avgpool1x1 = nn.AdaptiveAvgPool2d(1)

        self.prediction_head = nn.Linear(self.global_feature_dim, opt.class_num, bias=False)

    def forward(self, img_cat):
        '''  img_cat:b,4,256,256  '''
        global_feature = None
        if opt.without_mask:
            img_cat = img_cat[:,0:3] 
        feature_map = self.backbone(img_cat)  # b,512,8,8 (resnet layer4 output shape: b,c,8,8, if resnet18, c=512)
        global_feature = self.avgpool1x1(feature_map)  # b,512,1,1
        global_feature = global_feature.flatten(1) # b,512

        prediction = self.prediction_head(global_feature) 

        return prediction

if __name__ == '__main__':
    device = torch.device('cuda:0')
    b = 4
    img_cat = torch.randn(b, 4, 256, 256).to(device)
    model = ObjectPlaceNet(backbone_pretrained=False).to(device)
    local_pre = model(img_cat)
    print(local_pre)



