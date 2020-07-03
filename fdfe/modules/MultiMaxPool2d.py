import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiMaxPool2d(nn.Module):

    def __init__(self, 
                 maxpool2d):

        super(MultiMaxPool2d, self).__init__()
        self.maxpool2d = maxpool2d
        

    def forward(self, x):

        output = []

        x_ = F.pad(x, [0, 0, 0, 0], value=0)
        output.append(self.maxpool2d(x_))

        x_ = F.pad(x, [-1, 1, 0, 0], value=0)
        output.append(self.maxpool2d(x_))

        x_ = F.pad(x, [0, 0, -1, 1], value=0)
        output.append(self.maxpool2d(x_))

        x_ = F.pad(x, [-1, 1, -1, 1], value=0)
        output.append(self.maxpool2d(x_))

        return torch.cat(output, 0)