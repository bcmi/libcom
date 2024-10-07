import torch
import torch.nn as nn
import torchvision.models as models

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

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

class DoubleConv(BaseNetwork):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, norm_layer = nn.InstanceNorm2d):
        super().__init__()
        self.norm_layer = norm_layer
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            self.norm_layer(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            self.norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Bottleneck(BaseNetwork):
    def __init__(self,in_places,out_places,stride=1,downsampling=False,expansion=4, norm_layer = nn.InstanceNorm2d):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling
        self.norm_layer = norm_layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=out_places,kernel_size=1,stride=1),
            self.norm_layer(out_places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_places,out_channels=out_places,kernel_size=3,stride=stride,padding=1),
            self.norm_layer(out_places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_places,out_channels=out_places*self.expansion,kernel_size=1,stride=1),
            self.norm_layer(out_places*self.expansion)
        )

        if self.downsampling :
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places,out_channels=out_places*self.expansion,kernel_size=1,stride=stride),
                self.norm_layer(out_places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
class ResNet(BaseNetwork):
    def __init__(self, blocks, out_places=[64,16,32,64,64], input_nc = 4, expansion=4, norm_layer=nn.InstanceNorm2d):
        super(ResNet, self).__init__()
        self.expansion = expansion
        self.norm_layer = norm_layer
        self.bilinear = True 
        self.conv = DoubleConv(input_nc, 64, norm_layer=nn.BatchNorm2d)  #TODO:
        self.conv.eval()

        self.layer1 = self.make_layer(in_places=out_places[0], out_places=out_places[1], block=blocks[0], stride=2,
                                      norm_layer=self.norm_layer)
        self.layer2 = self.make_layer(in_places=4*out_places[1], out_places=out_places[2], block=blocks[1], stride=2,
                                      norm_layer=self.norm_layer)
        self.layer3 = self.make_layer(in_places=4*out_places[2], out_places=out_places[3], block=blocks[2], stride=2,
                                      norm_layer=self.norm_layer)
        self.layer4 = self.make_layer(in_places=4*out_places[3], out_places=out_places[4], block=blocks[3], stride=2,
                                      norm_layer=self.norm_layer)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def make_layer(self, in_places, out_places, block, stride, norm_layer=nn.InstanceNorm2d):
        layers = []
        layers.append(
            Bottleneck(in_places, out_places, stride, downsampling=True, norm_layer=norm_layer))
        for i in range(1, block):
            layers.append(
                Bottleneck(out_places * self.expansion, out_places, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, input):
        f1 = self.conv(input)
        f2 = self.layer1(f1)
        f3 = self.layer2(f2)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)
        return f5

class RegNetwork(BaseNetwork):

    def __init__(self,input_dim=256, norm_layer=nn.InstanceNorm2d):
        super(RegNetwork,self).__init__()
        self.norm_layer = norm_layer
        self.FeatureEncoder = ResNet([3,4,6,3],input_nc=4,norm_layer=self.norm_layer)
        self.net = nn.Sequential(
            nn.Conv2d(input_dim,input_dim*2,kernel_size=3,padding=1,stride=2),
            self.norm_layer(input_dim*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim*2,input_dim*2,kernel_size=3,padding=1,stride=1),
            self.norm_layer(input_dim*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim*2,input_dim*2,kernel_size=3,padding=1,stride=1),
            self.norm_layer(input_dim*2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.fc_bbx = nn.Linear(input_dim*2, 5)

    def forward(self, x):
        x = self.FeatureEncoder(x)
        x = self.net(x)
        pred_t = self.fc_bbx(x)
        pred_t[:, 4] = torch.tanh(pred_t[:, 4]) * (torch.pi / 2)
        return pred_t

class MaskCls(BaseNetwork):

    def __init__(self, num_classes=256):
        super(MaskCls, self).__init__()
        # 加载预训练的ResNet模型，这里选择ResNet50作为基础模型
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 替换最后一层全连接层
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        x = self.resnet(x)
        return x