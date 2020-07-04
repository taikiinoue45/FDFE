import copy
import torch
import torch.nn.functional as F

from tqdm import tqdm
from fdfe.models import Net, FDFENet

DEVICE = 'cuda:0'
H = 256
W = 512
img = torch.randn(1, 3, H, W).to(DEVICE)

patch_size = 64
padding_size = patch_size // 2
img = F.pad(img, [padding_size, padding_size, padding_size, padding_size], value=0)


net = Net().to(DEVICE)
net_copy = Net()
net_copy.load_state_dict(net.state_dict())
net_copy = net_copy.to(DEVICE)
print(torch.sum(net(img) == net_copy(img)))

fdfe = FDFENet(net_copy).to(DEVICE)
fdfe_out = fdfe(img)
fdfe_out = fdfe_out.detach().cpu()

fdfe_out_ = torch.zeros((1, 8, 32, 32, H, W))
for h in tqdm(range(H)):
    for w in range(W):
        patch = fdfe_out[:, :, h:h+32, w:w+32]
        fdfe_out_[:, :, :, :, h, w] = patch


net_out = torch.zeros((1, 8, 32, 32, H, W))
for h in tqdm(range(H)):
    for w in range(W):
        patch = img[:, :, h:h+patch_size, w:w+patch_size]
        patch = patch.to(DEVICE)
        patch_out = net(patch)
        patch_out = patch_out.detach().cpu()
        net_out[:, :, :, :, h, w] = patch_out

print(net_out.shape)
print(net_out.sum(), fdfe_out_.sum())
