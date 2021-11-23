from typing import Union
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as f

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.register_buffer('test', torch.tensor([1,2,3]))

    def forward(self, x):
        return x

net = MLP()
name = net.__getattr__('test')
print(name)