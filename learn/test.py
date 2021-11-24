from typing import Union
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as f

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.register_buffer('test', torch.tensor([1,2,3]))
        self.conv1 = nn.Conv2d(1,2,1)

    def forward(self, x):
        return x


net = MLP()

# torch.save(net.state_dict(), 'demo2.pth')
checkpoint = torch.load('demo2.pth')
net.load_state_dict()
state_dict = net.state_dict()
for data in state_dict._metadata.values():
    print(data)
    print(type(data))
# print(state_dict.__dict__)