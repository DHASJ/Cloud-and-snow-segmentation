import torch
from torch import nn


class iAFF2(nn.Module):
    def __init__(self, channels=64, out_channels=64, r=4, factors=1):
        super(iAFF2, self).__init__()
        inter_channels = int(channels // r)

        self.up = nn.UpsamplingBilinear2d(scale_factor=factors)

        # 本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0, groups=inter_channels),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), groups=inter_channels),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), groups=inter_channels),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0, groups=inter_channels),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        # 全局注意力
        self.conv_ = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.factors = factors
        self.final = nn.Sequential(
            nn.Conv2d(channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x, residual):
        if self.factors != 1:
            residual = self.up(residual)

        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        xg2 = self.conv_(result)
        xl2 = self.local_att2(x)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)

        output = self.final(xo)
        return output


if __name__ == '__main__':
    input1 = torch.randn(2, 64, 128, 128)
    input2 = torch.randn(2, 64, 64, 64)
    iaff = iAFF2(channels=64, out_channels=64, r=4, factors=2)
    print(iaff(input1, input2).shape)