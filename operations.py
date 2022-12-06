from torch import nn
import torch
import torch.nn.functional as F

OPNAMES = ["skip", "sep3x3", "sep5x5", "dil3x3", "dil5x5", "maxpool3x3", "avgpool3x3"]

class nconv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super().__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      padding=(kernel_size // 2), bias=False, stride=stride, groups=groups),
            nn.BatchNorm2d(out_channels, affine=True)
        )


class aconv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super().__init__(
            nconv(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=kernel_size, stride=stride, groups=groups),
            nn.ReLU()
        )


class zero(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mul(0.)


class reduced_zero(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def foward(self, x):
        return x.mul(0.)[:, :, x.shape[-2] // 2, x.shape[-1] // 2]


class sepconv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=kernel_size, padding=kernel_size//2, stride=stride, groups=in_channels, bias=False),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels, affine=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                      stride=1, padding=kernel_size//2, groups=in_channels, bias=False),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU()
        )


class dilconv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=2):
        super().__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size) - (dilation - 1), dilation=dilation, groups=in_channels, bias=False),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU()
        )


class separatedconv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, (1, kernel_size),
                      1, (kernel_size // 2, 0), bias=False),
            nn.Conv2d(in_channels, out_channels, (kernel_size, 1),
                      1, (0, kernel_size // 2), bias=False),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU()
        )


class skip(nn.Sequential):
    def __init__(self):
        super().__init__()

class MixedOp(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.candidates = nn.ModuleList()
        self.minit()

    def minit(self):
        self.candidates.append(skip())
        self.candidates.append(sepconv(self.channels, self.channels, 3, 1))
        self.candidates.append(sepconv(self.channels, self.channels, 5, 1))
        self.candidates.append(dilconv(self.channels, self.channels, 3, 1))
        self.candidates.append(dilconv(self.channels, self.channels, 5, 1))
        self.candidates.append(nn.MaxPool2d(3, 1, 1))
        self.candidates.append(nn.AvgPool2d(3, 1, 1))

    def forward(self, x, weights):
        return sum(w * m(x) for w, m in zip(weights, self.candidates))


class ReductionMixedOp(MixedOp):
    def __init__(self, channels, weights, contiguous=False):
        super().__init__(channels, weights)
        self.contiguous = contiguous