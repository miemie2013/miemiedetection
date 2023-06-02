import numpy as np
import torch
import torch.nn as nn


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(2)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


model = MyNet()

with torch.no_grad():
    weight_ = np.random.RandomState(113).randn(2, 3, 3, 3).astype(np.float32)
    model.conv.weight.copy_(torch.Tensor(weight_))
    bn_w = np.random.RandomState(114).randn(2).astype(np.float32)
    bn_b = np.random.RandomState(115).randn(2).astype(np.float32)
    model.bn.weight.copy_(torch.Tensor(bn_w))
    model.bn.bias.copy_(torch.Tensor(bn_b))
model = model.cuda()

need_clip = False
# need_clip = True
clip_norm = 0.1


'''
momentum = 0.9
lr = 0.9
weight_decay = 0.9999

# 不可以加正则化的参数：norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数。
no_L2, use_L2 = [], []
for name, param in model.named_parameters():
    # 只加入需要梯度的参数。
    if not param.requires_grad:
        continue
    if name.endswith('conv.weight'):
        use_L2.append(param)
    elif name.endswith('bn.weight'):
        no_L2.append(param)
    elif name.endswith('bn.bias'):
        no_L2.append(param)
    else:
        raise NotImplementedError("param name \'{}\' is not implemented.".format(name))
optimizer = torch.optim.SGD(no_L2, lr=lr, momentum=momentum)
for param_group in optimizer.param_groups:
    param_group["lr_factor"] = 1.0  # 设置 no_L2 的学习率
optimizer.add_param_group(
    {"params": use_L2, "weight_decay": weight_decay, "lr_factor": 1.0}
)
'''


lr = 0.9
weight_decay = 0.0001

# 不可以加正则化的参数：norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数。
no_L2, use_L2 = [], []
for name, param in model.named_parameters():
    # 只加入需要梯度的参数。
    if not param.requires_grad:
        continue
    if name.endswith('conv.weight'):
        use_L2.append(param)
    elif name.endswith('bn.weight'):
        no_L2.append(param)
    elif name.endswith('bn.bias'):
        no_L2.append(param)
    else:
        raise NotImplementedError("param name \'{}\' is not implemented.".format(name))
optimizer = torch.optim.AdamW(no_L2, betas=(0.9, 0.999), lr=lr, eps=1e-8, amsgrad=True)
for param_group in optimizer.param_groups:
    param_group["lr_factor"] = 1.0  # 设置 no_L2 的学习率
optimizer.add_param_group(
    {"params": use_L2, "weight_decay": weight_decay, "lr_factor": 1.0}
)


seed = 12
model.train()
for step_id in range(1):
    print('======================= step=%d ======================='%step_id)
    model.train()
    img = np.random.RandomState(seed).randn(2, 3, 4, 4).astype(np.float32)
    inputs = torch.Tensor(img).to(torch.float32).cuda()
    outs = model(inputs)
    print(outs)
    loss = outs.mean()

    loss.backward()
    if need_clip:
        for param_group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(param_group['params'], max_norm=clip_norm, norm_type=2)
    optimizer.step()
    optimizer.zero_grad()
    seed += 10

model.eval()
img = np.random.RandomState(seed+1).randn(2, 3, 4, 4).astype(np.float32)
inputs = torch.Tensor(img).to(torch.float32).cuda()
outs = model(inputs)

print('======================= eval =======================')
print(outs)

dic2 = np.load('001.npz')
outs2 = dic2['outs']

val33 = outs.cpu().detach().numpy()
ddd = np.sum((val33 - outs2) ** 2)
print('ddd val=%.6f' % ddd)


print('Done.')




