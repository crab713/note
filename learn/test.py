from typing import Union
import torch
from torch import Tensor
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.nn import functional as f

import os

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.register_buffer('test', torch.tensor([1,2,3]))
        self.conv1 = nn.Conv2d(1,2,1)

    def forward(self, x):
        return x

torch.cuda.set_device(-1)
torch.distributed.init_process_group(backend='nccl',
                                    world_size=1,
                                    init_method='env://',
                                    rank=0)

net = MLP()
