from torch import nn
import torch
from operations import OPNAMES, sepconv, skip, dilconv, aconv
import torch.nn.functional as F
from torch.autograd import Variable


class NormalCell(nn.Module):
    def __init__(self, channels, N=4, prev_reduced=False):
        super().__init__()
        self.N = N
        self.current = channels[2]
        if prev_reduced:
            self.preprocess0 = aconv(
                channels[0], channels[2], 1, 2, groups=2)  # factorized
        else:
            self.preprocess0 = aconv(channels[0], channels[2], 3)
        self.preprocess1 = aconv(channels[1], channels[2], 3)
        self.operators = nn.ModuleList()
        self.ginit()

    def ginit(self):
        self.edges = [-2, -1, -2, -1, -2, -1, -2, 0]
        self.operators.append(sepconv(self.current, self.current, 3))
        self.operators.append(sepconv(self.current, self.current, 3))
        self.operators.append(sepconv(self.current, self.current, 3))
        self.operators.append(sepconv(self.current, self.current, 3))
        self.operators.append(skip())
        self.operators.append(sepconv(self.current, self.current, 3))
        self.operators.append(skip())
        self.operators.append(dilconv(self.current, self.current, 3))

    def forward(self, x, y, alphas):
        states = [self.preprocess0(x), self.preprocess1(y)]
        for i, l in enumerate(self.edges):
            if i % 2 == 0:
                if i != 0:
                    states.append(s)
                s = self.operators[i](states[l + 2])
            else:
                s = s + self.operators[i](states[l + 2])  # ! no inplace please
                if i == len(self.edges) - 1:
                    states.append(s)
        return torch.cat(states[2:], dim=1)


class ReductionCell(NormalCell):
    def __init__(self, channels, N=4, prev_reduced=False):
        super().__init__(channels, N, prev_reduced)

    def ginit(self):
        self.edges = [-2, -1, -1, 0, -2, 0, -1, 0]
        self.operators.append(nn.MaxPool2d(3, 2, 1))
        self.operators.append(nn.MaxPool2d(3, 2, 1))
        self.operators.append(nn.MaxPool2d(3, 2, 1))
        self.operators.append(skip())
        self.operators.append(nn.MaxPool2d(3, 2, 1))
        self.operators.append(skip())
        self.operators.append(nn.MaxPool2d(3, 2, 1))
        self.operators.append(skip())


class NASNet(nn.Module):
    def __init__(self, device=torch.device('mps'), N=4, L=8, C=32, A=len(OPNAMES)):
        super().__init__()
        self.alphas = Variable(torch.zeros(2, N * (N + 3) // 2, A, device=device), requires_grad=True)
        self.N = N
        self.prev0 = 3
        self.prev1 = 3
        self.current = C
        self.cells = nn.ModuleList()
        reduced = 0
        for i in range(0, L):
            if i % ((L + 2) // 3) == 0 and i != 0:
                self.current *= 2
                self.cells.append(self.getReductionCell(
                    [self.prev0, self.prev1, self.current], self.N))
                reduced = 1
            else:
                self.cells.append(self.getNormalCell(
                    channels=[self.prev0, self.prev1, self.current], N=self.N, prev_reduced=reduced))
                reduced = 0
            print((self.prev0, self.prev1, self.current))
            self.prev0 = self.prev1
            self.prev1 = self.current * N

        self.reduction = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.prev1, 10)

    def getReductionCell(self, channels, N):
        return ReductionCell(channels, N)

    def getNormalCell(self, channels, N, prev_reduced):
        return NormalCell(channels, N, prev_reduced)

    def forward(self, x):
        x = x
        y = x
        weights = F.softmax(self.alphas[0], dim=-1)
        for cell in self.cells:
            z = cell(x, y, weights)
            x = y
            y = z
        y = self.reduction(y)
        y = y.view(y.shape[0], -1)
        return self.classifier(y)
