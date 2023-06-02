import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import ResNet
from mmdet.utils import load_ckpt


class MyNet(nn.Module):
    def __init__(self, backbone):
        super(MyNet, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        outs = self.backbone(x)
        return outs

backbone_ = dict(
    depth=50,
    variant='d',
    return_idx=[1, 2, 3],
    dcn_v2_stages=[3],
    freeze_at=-1,
    freeze_norm=False,
    norm_decay=0.,
)

backbone = ResNet(**backbone_)
model = MyNet(backbone)

param_path = '../ppyolov2_r50vd_365e.pth'
ckpt = torch.load(param_path, map_location="cpu")["model"]
model = load_ckpt(model, ckpt)


momentum = 0.9
lr = 0.005
weight_decay = 0.0005

# 不可以加正则化的参数：norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数。
no_L2, use_L2 = [], []
for name, param in model.named_parameters():
    # 只加入需要梯度的参数。
    if not param.requires_grad:
        continue
    if name.startswith('backbone.'):
        if name.endswith('.conv.weight'):
            use_L2.append(param)
        elif name.endswith('.norm.weight'):
            no_L2.append(param)
        elif name.endswith('.norm.bias'):
            no_L2.append(param)
        elif name.endswith('.conv_offset.weight'):  # 可变形卷积的conv_offset.weight, 需要L2
            use_L2.append(param)
        elif name.endswith('.conv_offset.bias'):  # 可变形卷积的conv_offset.bias,   需要L2
            use_L2.append(param)
        else:
            raise NotImplementedError("param name \'{}\' is not implemented.".format(name))
    else:
        raise NotImplementedError("param name \'{}\' is not implemented.".format(name))
optimizer = torch.optim.SGD(no_L2, lr=lr, momentum=momentum)
for param_group in optimizer.param_groups:
    param_group["lr_factor"] = 1.0  # 设置 no_L2 的学习率
optimizer.add_param_group(
    {"params": use_L2, "weight_decay": weight_decay, "lr_factor": 1.0}
)

seed = 12
model.cuda()
model.train()
for step_id in range(8):
    print('======================= step=%d ======================='%step_id)
    model.train()
    img = np.random.RandomState(seed).randn(2, 3, 640, 640)
    inputs = torch.Tensor(img).to(torch.float32).cuda()
    outs = model(inputs)
    loss = outs[0].mean() + outs[1].mean() + outs[2].mean()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    seed += 10

model.eval()
img = np.random.RandomState(seed+1).randn(2, 3, 640, 640)
inputs = torch.Tensor(img).to(torch.float32).cuda()
outs = model(inputs)

print('======================= eval =======================')

dic2 = np.load('002.npz')
outs2 = dic2['outs2']

val33 = outs[2].cpu().detach().numpy()
ddd = np.mean((val33 - outs2) ** 2)
print('ddd val=%.6f' % ddd)


print('Done.')




