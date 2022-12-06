from torch import nn
from operations import aconv, nconv
import torch

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = aconv(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride)
        self.conv2 = nconv(in_channels=out_channels,
                           out_channels=out_channels, kernel_size=kernel_size)
        self.conv3 = nconv(in_channels=out_channels,
                           out_channels=out_channels, kernel_size=kernel_size + 2)
        self.conv4 = nconv(in_channels=out_channels,
                           out_channels=out_channels, kernel_size=kernel_size + 4)
        self.proj = nn.Sequential()
        if in_channels != out_channels:
            self.proj = nconv(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        self.relu = nn.ReLU()
        self.coefficient = nn.parameter.Parameter(torch.tensor([1.0, 1.0, 1.0]))

    def forward(self, x):
        out = self.conv1(x)
        out1 = self.conv2(out)
        out2 = self.conv3(out)
        out3 = self.conv4(out)
        x = self.proj(x)
        return self.relu(self.coefficient[0] * out1 + self.coefficient[1] * out2 + self.coefficient[2] * out3 + x)


class ResNetConv(nn.Sequential):
    def __init__(self):
        channels = [3, 64, 128, 256, 512]
        super().__init__(
            ResBlock(in_channels=channels[0], out_channels=channels[1]),
            ResBlock(in_channels=channels[1], out_channels=channels[2]),
            ResBlock(in_channels=channels[2], out_channels=channels[3]),
            nn.MaxPool2d(2),
            *(4 * [ResBlock(in_channels=channels[3], out_channels=channels[3])]),
            ResBlock(in_channels=channels[3], out_channels=channels[4]),
            ResBlock(in_channels=channels[4], out_channels=channels[4]),

            nn.AdaptiveMaxPool2d((1, 1))
        )


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ResNetConv()
        self.linear = nn.Linear(512, 10)
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 512)
        return self.linear(x)