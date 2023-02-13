import torch
from torch import nn


class CGCAM(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CGCAM, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(in_dim//2, in_dim//2, 1, groups=in_dim//2)
        self.sigmoid = nn.Sigmoid()
        self.in_dim = in_dim
        self.conv = nn.Conv2d(in_dim, out_dim, 1)

    def forward(self, x):
        x = self.gap(x)
        [x1, x2] = torch.split(x, self.in_dim//2, 1)
        x1 = self.conv1x1(x1)
        x2 = self.conv1x1(x2)

        x1 = x1 + x2
        x1 = self.conv1x1(x1)

        x2 = x1 + x2
        x2 = self.conv1x1(x2)

        x1_out = self.sigmoid(x1)
        x2_out = self.sigmoid(x2)
        output = torch.cat([x1_out, x2_out], dim=1)
        output = self.conv(output)
        return output