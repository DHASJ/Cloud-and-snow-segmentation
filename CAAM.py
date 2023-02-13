import torch
from torch import nn


class Spatical(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Spatical, self).__init__()
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//2, 1, groups=out_dim//2),
            nn.BatchNorm2d(out_dim//2),
            nn.GELU()
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 1, groups=out_dim),
            nn.BatchNorm2d(out_dim),
            nn.GELU()
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(out_dim//2, out_dim, 1, groups=out_dim//2),
            nn.BatchNorm2d(out_dim),
            nn.GELU()
        )
        self.maxpool = nn.MaxPool2d(3, 1, 1)
        self.avgpool = nn.AvgPool2d(3, 1, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.dwconv1(x)
        x_ = self.dwconv(x)
        b, c, h, w = x_.shape
        x__ = x_.view(b, c, -1)

        max = self.maxpool(x)
        avg = self.avgpool(x)
        x1 = torch.cat([max, avg], dim=1)
        x1 = self.dwconv2(x1)
        x1 = x1.view(b, c, -1)
        x1 = self.softmax(x1)
        x1 = x1 * x__

        output = x1.view(b, c, h, w)
        output = output + x_

        return output


class Channel(nn.Module):
    def __init__(self, dim, out_dim):
        super(Channel, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.convl_1 = nn.Conv2d(dim, dim//4, 1, groups=dim//4)
        self.convl_2 = nn.Conv2d(dim//4, dim, 1, groups=dim//4)

        self.convr_1 = nn.Conv2d(dim, dim * 4, 1, groups=dim)
        self.convr_2 = nn.Conv2d(dim * 4, dim, 1, groups=dim)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(dim, out_dim, 1, groups=out_dim)

    def forward(self, x):
        x = self.gap(x)
        left = self.convl_1(x)
        left = self.convl_2(left)
        left = self.sigmoid(left)
        right = self.convr_1(x)
        right = self.convr_2(right)
        right = self.sigmoid(right)
        output = left + right
        output = self.conv(output)
        return output


class SC(nn.Module):
    def __init__(self, factor, in_dim1, in_dim2):  # in_dim1:浅    in_dim2:深
        super(SC, self).__init__()
        self.channel = Channel(in_dim1+in_dim2, in_dim1)
        self.spatical = Spatical(in_dim1, in_dim1)
        self.conv = nn.Conv2d(in_dim2, in_dim1, 1, groups=in_dim1)
        self.up = nn.Upsample(scale_factor=factor)

    def forward(self, x1, x2):
        x2__ = self.up(x2)
        x2 = torch.cat([x1, x2__], dim=1)
        x2 = self.channel(x2)
        x1 = x1*x2
        x1 = self.spatical(x1)
        x2__ = self.conv(x2__)
        output = x1 + x2__
        return output


if __name__ == '__main__':
    input1 = torch.randn(2, 512, 128, 128)
    input2 = torch.randn(2, 512, 128, 128)
    sc = SC(factor=1, in_dim1=512, in_dim2=512)
    print(sc(input1, input2).shape)
