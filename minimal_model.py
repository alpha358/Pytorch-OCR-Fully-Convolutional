import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np

from torch.nn.modules.normalization import LayerNorm,LocalResponseNorm
from torch.nn import BatchNorm2d,MaxPool2d




class MinimalModel(nn.Module):
    '''
    Input[Batch, Vertical, Horisontal, Chnl]

    Horizontal dim becomes character dimension.
    Average over vertical dimension.
    '''

    def __init__(self,
                nclasses
                ):
        super(SimpleModel, self).__init__()

        self.classes=nclasses
        
        # in_channels, out_channels, kernel_size, stride=1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)

        self.elu=nn.ELU()
        self.max_pool=nn.MaxPool2d()


        
    def forward(self, x):
        '''
        x --- [batch, width, height, chnl]
        '''
        x = self.conv1(x)
        x = self.elu(x)
        # 

        x = self.conv2(x)
        x = self.elu(x)
        # 

        x = self.conv3(x)
        x = self.elu(x)

        # average pool the height dimension
        x=torch.mean(x,dim=2) # or max ?
        x=nn.LogSoftmax(dim=1)(x)
        
