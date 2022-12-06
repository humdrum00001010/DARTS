import torch
import torch.nn as nn
# import torch.nn.functional as F
from operations import MixedOp
from darts import NASNet, NormalCell

class TrainCell(NormalCell):
    def __init__(self, channels, N=4, prev_reduced=False):
        super().__init__(channels, N, prev_reduced)
    def ginit(self):
        edge_count = self.N * (self.N + 3) // 2
        for i in range(0, edge_count):
            self.operators.append(MixedOp(self.current))
    def forward(self, x, y, alphas):
        states = [self.preprocess0(x), self.preprocess1(y)]
        '''
        for i in range(0, self.N): # number of nodes.
            for j in range(0, i + 2): # number of directed edge to the node i.
                idx = i * (i + 3) // 2 + j
                edge = self.operators[idx]
                if j == 0:
                    current = edge(states[j], alphas[idx])
                else:
                    current = current + edge(states[j], alphas[idx])
                if j == i + 1:
                    states.append(current)
        '''
        for i in range(0, self.N):
            s = sum(self.operators[i * (i + 3) // 2 + j](s0, alphas[i * (i + 3) // 2 + j]) for j, s0 in enumerate(states))
            states.append(s)
        return torch.cat(states[2:], dim=1)


class TrainNASNet(NASNet):
    def __init__(self, device=torch.device('mps'), N = 4, L = 8, C = 32):
        super().__init__(device, N, L, C)
    def getNormalCell(self, channels, N, prev_reduced):
        return TrainCell(channels=channels, N=self.N, prev_reduced=prev_reduced)
    def getReductionCell(self, channels, N):
        return super().getReductionCell(channels, N)