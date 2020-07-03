import torch
import torch.nn as nn


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        layers = [nn.Conv2d(1, 8, kernel_size=3, stride=1),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2),
                  nn.Conv2d(8, 16, kernel_size=3, stride=1),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2),
                  nn.Conv2d(16, 128, kernel_size=2, stride=1),
                  nn.ReLU(),
                  nn.Conv2d(128, 113, kernel_size=1)]
        self.layers = nn.ModuleList(layers)


    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        return x