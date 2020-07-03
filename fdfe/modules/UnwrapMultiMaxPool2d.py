import torch.nn as nn

class UnwrapMultiMaxPool2d(nn.Module):

    def __init__(self):
        
        super(UnwrapMultiMaxPool2d, self).__init__()
        self.H = 960
        self.W = 1280


    def forward(self, x):

        x = x.view(4, -1).contiguous()
        x = x.transpose(0, 1).contiguous()
        x = x.view(8, self.H//2, self.W//2, 2, 2).contiguous()
        x = x.transpose(2, 3).contiguous()
        x = x.view(8, self.H, self.W).contiguous()
        return x
