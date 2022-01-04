import torch
import numpy as np
import torch.nn.functional as F

'''

'''



dic2 = np.load('ClipGradByGlobalNorm.npz')



x = dic2['x']
w = dic2['w']
b = dic2['b']
loss_2 = dic2['loss']
w_2 = dic2['w_2']
b_2 = dic2['b_2']


x = torch.from_numpy(x)


linear = torch.nn.Linear(in_features=10, out_features=2)
w = w.transpose(1, 0)
linear.weight.data = torch.from_numpy(w)
linear.bias.data = torch.from_numpy(b)

# 建立优化器
scaler = torch.cuda.amp.GradScaler(enabled=False)  # 不使用混合精度训练
param_groups = []
base_lr = 0.1
param_group_conv = {'params': [linear.weight]}
param_group_conv['lr'] = base_lr
param_group_conv['need_clip'] = True
param_group_conv['clip_norm'] = 1.0
# param_group_conv['weight_decay'] = base_wd
param_groups.append(param_group_conv)
if linear.bias is not None:
    if linear.bias.requires_grad:
        param_group_conv_bias = {'params': [linear.bias]}
        param_group_conv_bias['lr'] = base_lr
        param_group_conv_bias['need_clip'] = False
        # param_group_conv_bias['weight_decay'] = 0.0
        param_groups.append(param_group_conv_bias)

optimizer = torch.optim.SGD(
    param_groups, lr=base_lr
)


# 计算损失
out = linear(x)
loss = 1000.0 * torch.mean(out)  # 乘以1000.0放大损失，使得梯度的模很大



optimizer.zero_grad()
scaler.scale(loss).backward()
# 梯度裁剪
for param_group in optimizer.param_groups:
    if param_group['need_clip']:
        torch.nn.utils.clip_grad_norm_(param_group['params'], max_norm=param_group['clip_norm'], norm_type=2)
scaler.step(optimizer)
scaler.update()


www2 = linear.weight.data.cpu().detach().numpy()
www2 = www2.transpose(1, 0)
bbb2 = linear.bias.data.cpu().detach().numpy()

ddd = np.sum((w_2 - www2)**2)
print('ddd=%.6f' % ddd)
ddd = np.sum((b_2 - bbb2)**2)
print('ddd=%.6f' % ddd)
print()









