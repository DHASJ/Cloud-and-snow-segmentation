import torch
from torch import nn

from backbone import resnet50, resnet18
from backbone2 import resnet18_2
from CAAM import SC
from MSSCMs import De
from MSSCM import convfeiduicheng
from CGCAM import CGCAM


class mynet13_2_last(nn.Module):
    def __init__(self, num_classes=3, backbone='resnet18'):
        super(mynet13_2_last, self).__init__()
        if backbone == 'myresnet':
            self.backbone = resnet18_2()
            filters = [32, 64, 64, 128, 256, 512]
        elif backbone == 'resnet50':
            self.backbone = resnet50(pretrained=True)
            filters = [32, 64, 256, 512, 1024, 2048]
        elif backbone == 'resnet18':
            self.backbone = resnet18(pretrained=True)
            filters = [32, 64, 64, 128, 256, 512]

        self.sc1 = SC(factor=2, in_dim1=filters[2], in_dim2=filters[3])
        self.sc2 = SC(2, filters[3], filters[4])
        self.sc3 = SC(2, filters[4], filters[5])

        self.duo1 = convfeiduicheng(filters[1] + filters[1], filters[2])
        self.duo2 = convfeiduicheng(filters[3], filters[2])
        self.duo3 = convfeiduicheng(filters[4], filters[3])
        self.duo4 = nn.Sequential(
            convfeiduicheng(filters[4], filters[3]),
            # nn.UpsamplingBilinear2d(scale_factor=2),
            # nn.Conv2d(filters[4], filters[3], 1, groups=filters[3])
        )
        # self.duo1 = nn.Conv2d(filters[1]+filters[1], filters[2], 1)
        # self.duo2 = nn.Conv2d(filters[3], filters[2], 1)
        # self.duo3 = nn.Conv2d(filters[4], filters[3], 1)
        # self.duo4 = nn.Sequential(
        #     # nn.Conv2d(filters[5], filters[4], 1),
        #     nn.Conv2d(filters[4], filters[3], 1, groups=filters[3])
        # )

        self.iaff1 = iAFF2(filters[2], filters[0], factors=2)
        self.iaff2 = iAFF2(filters[2], filters[2], factors=2)
        self.iaff3 = iAFF2(filters[3], filters[2], factors=2)

        self.non = CGCAM(filters[5], filters[4])

        self.up1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(filters[3], filters[1], 1)
        )
        self.up2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(filters[4], filters[2], 1)
        )

        self.up3 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(filters[4], filters[3], 1)
        )

        self.up4 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(filters[5], filters[4], 1)
        )

        self.up5 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(filters[0], filters[0], 1)
        )
        self.conv = nn.Sequential(
            # nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(filters[1]+filters[1], filters[0], 1, groups=filters[0]),
            nn.BatchNorm2d(filters[0]),
            nn.GELU()
        )
        self.de = De(filters[0], filters[0] // 2)
        # self.de = nn.Conv2d(filters[0], filters[0]//2, 1)
        self.final = nn.Conv2d(filters[0] // 2, num_classes, 1)

    def forward(self, x):
        [x1, x2, x3, x4, x5] = self.backbone(x)
        sc1 = self.sc1(x2, x3)
        sc2 = self.sc2(x3, x4)
        sc3 = self.sc3(x4, x5)
        x5 = self.non(x5)
        # x5 = self.up4(x5)

        sc3 = x5 * sc3
        sc3_ = self.up3(sc3)
        duo4 = self.duo4(sc3)

        sc2 = torch.cat([sc3_, sc2], dim=1)
        sc2_ = self.up2(sc2)
        duo3 = self.duo3(sc2)

        sc1 = torch.cat([sc2_, sc1], dim=1)
        sc1_ = self.up1(sc1)
        duo2 = self.duo2(sc1)

        x1 = torch.cat([x1, sc1_], dim=1)
        x1_ = self.conv(x1)
        duo1 = self.duo1(x1)

        iaff3 = self.iaff3(duo3, duo4)
        iaff2 = self.iaff2(duo2, iaff3)
        iaff1 = self.iaff1(duo1, iaff2)

        output = x1_ + iaff1
        output = self.up5(output)

        output = self.de(output)
        output = self.final(output)

        return output