import torch
from torch import nn
import model
import torch.nn.functional as F

img = torch.zeros(3, 32)

print([m for m in F.softmax(img, dim=-1)])