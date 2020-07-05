import copy
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from fdfe.models import Net, FDFENet

DEVICE = 'cuda:0'
H = 256
W = 512
PATCH_SIZE = 64


img = torch.randn(1, 3, H, W).to(DEVICE)
img = F.pad(img, [32, 31, 32, 31], value=0)


net = Net().to(DEVICE)
net_copy = Net()
net_copy.load_state_dict(net.state_dict())

net_map = np.zeros((H, W))
for h in tqdm(range(H)):
    for w in range(W):
        patch = img[:, :, h:h+PATCH_SIZE, w:w+PATCH_SIZE]
        patch = patch.to(DEVICE)
        net_out = net(patch)
        net_out = net_out.mean()
        net_out = net_out.detach().cpu()
        net_map[h, w] = float(net_out)


fdfe = FDFENet(net_copy).to(DEVICE)
fdfe_out = fdfe(img)
fdfe_out = fdfe_out.detach().cpu()

b_, c_, h_, w_ = net(patch).shape
fdfe_map = np.zeros((H, W))
for h in tqdm(range(H)):
    for w in range(W):

        if h % 2 == 0 and w % 2 == 0:
            fdfe_map[h, w] = float(fdfe_out[0, :, h//2:h//2+h_, w//2:w//2+w_].mean())
        elif h % 2 == 0 and w % 2 == 1:
            fdfe_map[h, w] = float(fdfe_out[1, :, h//2:h//2+h_, w//2:w//2+w_].mean())
        elif h % 2 == 1 and w % 2 == 0:
            fdfe_map[h, w] = float(fdfe_out[2, :, h//2:h//2+h_, w//2:w//2+w_].mean())
        elif h % 2 == 1 and w % 2 == 1:
            fdfe_map[h, w] = float(fdfe_out[3, :, h//2:h//2+h_, w//2:w//2+w_].mean())

error = np.abs(fdfe_map - net_map).sum() / np.abs(fdfe_map).sum()
print(f'error: {error}')