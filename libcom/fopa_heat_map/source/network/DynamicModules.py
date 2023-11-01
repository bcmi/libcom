import torch.nn as nn
import torch.nn.functional as F


class simpleDFN(nn.Module):
    def __init__(self, in_xC, in_yC, out_C, kernel_size=3, down_factor=4):
        """use nn.Unfold to realize dynamic convolution

        Args:
            in_xC (int): channel number of first input
            in_yC (int): channel number of second input
            out_C (int): channel number of output
            kernel_size (int): the size of generated conv kernel
            down_factor (int): reduce the model parameters when generating conv kernel
        """
        super(simpleDFN, self).__init__()
        self.kernel_size = kernel_size
        self.fuse = nn.Conv2d(in_xC, out_C, 3, 1, 1)
        self.out_C = out_C
        self.gernerate_kernel = nn.Sequential(
            # nn.Conv2d(in_yC, in_yC, 3, 1, 1),
            # DenseLayer(in_yC, in_yC, k=down_factor),
            nn.Conv2d(in_yC, in_xC, 1),
        )
        self.unfold = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)
        self.pool = nn.AdaptiveAvgPool2d(self.kernel_size)
        self.in_planes = in_yC

    def forward(self, x, y):  # x:bg y:fg
        kernel = self.gernerate_kernel(self.pool(y))
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)
        kernel = kernel.view(-1, 1, self.kernel_size, self.kernel_size)
        if self.kernel_size == 3:
            output = F.conv2d(x, kernel, bias=None, stride=1, padding=1, groups=self.in_planes * batch_size)
        elif self.kernel_size == 1:
            output = F.conv2d(x, kernel, bias=None, stride=1, padding=0, groups=self.in_planes * batch_size)
        elif self.kernel_size == 5:
            output = F.conv2d(x, kernel, bias=None, stride=1, padding=2, groups=self.in_planes * batch_size)
        else:
            output = F.conv2d(x, kernel, bias=None, stride=1, padding=3, groups=self.in_planes * batch_size)
        output = output.view(batch_size, -1, height, width)
        return self.fuse(output)