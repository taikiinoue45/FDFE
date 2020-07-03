import tqdm
import torch

from fdfe.models import Net, FDFENet

H = 960
W = 1280
img = torch.randn(1, 1, H, W)

net = Net()
fdfe_net = FDFENet(net)

out = fdfe_net(img)
print(out.shape)