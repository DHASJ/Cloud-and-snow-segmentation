import torch
from torch import nn


class De(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(De, self).__init__()
        self.dwconv_3x3 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, 1, 1, groups=out_dim),
            nn.BatchNorm2d(out_dim),
            nn.GELU()
        )
        self.conv_1x3_3x1 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), groups=out_dim),
            nn.Conv2d(out_dim, out_dim, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), groups=out_dim),
            nn.BatchNorm2d(out_dim),
            nn.GELU()
        )
        self.conv_1x5_5x1 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), groups=out_dim),
            nn.Conv2d(out_dim, out_dim, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), groups=out_dim),
            nn.BatchNorm2d(out_dim),
            nn.GELU()
        )
        self.conv_1x7_7x1 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), groups=out_dim),
            nn.Conv2d(out_dim, out_dim, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), groups=out_dim),
            nn.BatchNorm2d(out_dim),
            nn.GELU()
        )
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(out_dim*3, out_dim, 1, groups=out_dim),
            nn.BatchNorm2d(out_dim),
            nn.GELU()
        )

    def forward(self, x):
        x = self.dwconv_3x3(x)
        x1 = self.conv_1x3_3x1(x)
        x2 = self.conv_1x5_5x1(x)
        x3 = self.conv_1x7_7x1(x)
        output = torch.cat([x1, x2, x3], dim=1)
        output = self.conv_1x1(output)
        output = output + x

        return output


