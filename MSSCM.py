import torch
from torch import nn


class convfeiduicheng(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(convfeiduicheng, self).__init__()
        self.conv1x3dilation = nn.Sequential(
            nn.Conv2d(in_dim // 2, in_dim // 2, (3, 1), (1, 1), dilation=(2, 1),  padding=(2, 0),groups=in_dim // 2),
            nn.Conv2d(in_dim // 2, in_dim // 2, (1, 3), (1, 1), dilation=(1, 2), padding=(0, 2), groups=in_dim // 2),
            nn.Conv2d(in_dim // 2, in_dim // 2, (3, 1), (1, 1), dilation=(2, 1), padding=(2, 0), groups=in_dim // 2),
            nn.Conv2d(in_dim // 2, in_dim // 2, (1, 3), (1, 1), dilation=(1, 2), padding=(0, 2), groups=in_dim // 2)
        )
        self.conv1x5dilation = nn.Sequential(
            nn.Conv2d(in_dim // 2, in_dim // 2, (1, 5), (1, 1), dilation=(1, 2), padding=(0, 4), groups=in_dim // 2),
            nn.Conv2d(in_dim // 2, in_dim // 2, (5, 1), (1, 1), dilation=(2, 1),  padding=(4, 0),groups=in_dim // 2),
            nn.Conv2d(in_dim // 2, in_dim // 2, (1, 5), (1, 1), dilation=(1, 2), padding=(0, 4), groups=in_dim // 2),
            nn.Conv2d(in_dim // 2, in_dim // 2, (5, 1), (1, 1), dilation=(2, 1), padding=(4, 0), groups=in_dim // 2)
        )
        self.in_dim = in_dim
        self.dim = min(in_dim, out_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, groups=self.dim),
            nn.BatchNorm2d(out_dim),
            nn.GELU()
        )

    def forward(self, x):
        [x1, x2] = torch.split(x, self.in_dim//2, 1)
        x1 = self.conv1x5dilation(x1)
        x2 = self.conv1x3dilation(x2)
        output = torch.cat([x1, x2], dim=1)
        output = output + x
        output = self.conv(output)
        return output

if __name__ == '__main__':
    