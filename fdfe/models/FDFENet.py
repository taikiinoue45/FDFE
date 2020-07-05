import torch
import torch.nn as nn
from fdfe.modules import MultiMaxPool2d, UnwrapMultiMaxPool2d


class FDFENet(nn.Module):
    
    def __init__(self,
                 net: nn.Module):
        
        super(FDFENet, self).__init__()
        
        for i, layer in enumerate(net.layers):
            if layer._get_name() == 'MaxPool2d':
                net.layers[i] = MultiMaxPool2d(layer)
                
        self.fdfe_net = net
        
        
    def forward(self, x):
        
        x = self.fdfe_net(x)
        return x