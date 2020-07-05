import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiMaxPool2d(nn.Module):

    def __init__(self, 
                 maxpool2d: nn.Module):
        
        '''
        A ｍodule to perform all the possible MaxPool2d computations.
        In kernel_size=2 and stride=2, four different MaxPool2d computations are possible.
        Currently, this module only supports nn.MaxPool2d with kernel_size=2 and stride=2.
        '''

        super(MultiMaxPool2d, self).__init__()

        kernel_size = maxpool2d.kernel_size
        stride = maxpool2d.stride
        if kernel_size != 2 or stride != 2:
            print('Currently, MultiMaxPool2d only supports nn.MaxPool2d with kernel_size=2 and stride=2.')
            sys.exit()

        self.maxpool2d = maxpool2d
        

    def forward(self, 
                x: torch.tensor) -> torch.tensor:

        out = []
        
        x_ = F.pad(x, [0, 0, 0, 0], value=0)
        out.append(self.maxpool2d(x_))

        x_ = F.pad(x, [-1, 1, 0, 0], value=0)
        out.append(self.maxpool2d(x_))

        x_ = F.pad(x, [0, 0, -1, 1], value=0)
        out.append(self.maxpool2d(x_))

        x_ = F.pad(x, [-1, 1, -1, 1], value=0)
        out.append(self.maxpool2d(x_))

        return torch.cat(out, 0)
