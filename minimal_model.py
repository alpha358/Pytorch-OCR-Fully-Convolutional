import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np

from torch.nn.modules.normalization import LayerNorm,LocalResponseNorm
from torch.nn import BatchNorm2d,MaxPool2d




class MinimalModel(nn.Module):
    def __init__(self,
                nclasses
                ):
        super(SimpleModel, self).__init__()

        self.classes=nclasses
        self.reduce1 = nn.Conv2d(3, 16, kernel_size=1)


        
    def forward(self, x):
        '''
        x --- [batch, width, height]
        '''
        x=self.reduce1(x)
        x=nn.LogSoftmax(dim=1)(x)
        
