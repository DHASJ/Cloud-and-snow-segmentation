import torch
import torch.nn as nn

def dwconv_1x1(in_dim, out_dim, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, groups=groups),
        nn.BatchNorm2d(out_dim),
        nn.GELU()
    )


def dwconv_3x3(in_dim, out_dim, stride=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 3, stride, 1, groups=groups),
        nn.BatchNorm2d(out_dim),
        nn.GELU()
    )


class my_Blocks(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(my_Blocks, self).__init__()
        self.dwconv_1x1_1 = dwconv_1x1(inplanes, planes, groups=planes)

        self.dwconv_3x3_1 = dwconv_3x3(planes, planes, stride=stride, groups=planes)
        self.dwconv_3x3_2 = dwconv_3x3(planes, planes, stride=1, groups=planes)

        self.dwconv_1x1_2 = dwconv_1x1(planes, planes*self.expansion, groups=planes)

        self.down = nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, 2, 1, groups=planes)
        )
        self.downsample = downsample
        self.relu = nn.GELU()

    def forward(self, x):
        identity = x

        x_ = self.dwconv_1x1_1(x)

        middle = self.dwconv_3x3_1(x_)
        left = self.dwconv_3x3_2(middle)
        left = self.dwconv_3x3_2(left)
        right = left + middle
        right = self.dwconv_3x3_2(right)

        if self.downsample is not None:
            x_ = self.down(x)

        output = middle + left + right + x_
        output = self.dwconv_1x1_2(output)

        if self.downsample is not None:
            identity = self.downsample(x)

        output = identity + output
        output = self.relu(output)

        return output


def conv3x1_1x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0), bias=False),
        nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1), bias=False)
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck_2(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d, relu=nn.GELU):
        super(Bottleneck_2, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        if stride != 1:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        else:
            self.conv2 = conv3x1_1x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = relu()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, norm_layer=nn.BatchNorm2d, relu=nn.GELU):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = relu()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the TNet by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck_2):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, my_Blocks):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                # conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.Conv2d(self.inplanes, planes * block.expansion, 3, stride, padding=1, groups=self.inplanes),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)

        x = self.maxpool(x1)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        # x = self.avgpool(x5)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x1, x2, x3, x4, x5


def resnet18_2(**kwargs):
    model = ResNet(my_Blocks, [2, 2, 2, 2], **kwargs)
    return model


if __name__ == '__main__':
    x = torch.randn(4, 3, 256, 256)
    resnet50 = resnet18_2()
    print(
        resnet50(x)[0].shape,  # torch.Size([4, 64, 128, 128])
        resnet50(x)[1].shape,  # torch.Size([4, 64, 64, 64])
        resnet50(x)[2].shape,  # torch.Size([4, 128, 32, 32])
        resnet50(x)[3].shape,  # torch.Size([4, 256, 16, 16])
        resnet50(x)[4].shape,  # torch.Size([4, 512, 8, 8])
        )