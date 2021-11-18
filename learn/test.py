from torch import nn
import torch
from torch.autograd import Function

x = torch.ones(8,3,20)
mean_val = x.mean([0,2])
x = x - mean_val[None, ..., None]
print(x.shape)