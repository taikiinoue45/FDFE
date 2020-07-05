import torch
import torch.nn as nn


class Net(nn.Module):
    
    def __init__(self):

        super(Net, self).__init__()
        
        layers = [nn.Conv2d(3, 8, kernel_size=3, stride=1),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2)]
        self.layers = nn.ModuleList(layers)


    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        return x