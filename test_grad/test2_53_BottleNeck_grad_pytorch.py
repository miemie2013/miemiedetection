
import torch
import numpy as np
from mmdet.models import BottleNeck


ch_in = 64
ch_out = 64
stride = 1
shortcut = False
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 256
ch_out = 64
stride = 1
shortcut = True
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 256
ch_out = 64
stride = 1
shortcut = True
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 256
ch_out = 128
stride = 2
shortcut = False
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 512
ch_out = 128
stride = 1
shortcut = True
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 512
ch_out = 128
stride = 1
shortcut = True
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 512
ch_out = 128
stride = 1
shortcut = True
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 512
ch_out = 256
stride = 2
shortcut = False
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 1024
ch_out = 256
stride = 1
shortcut = True
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 1024
ch_out = 256
stride = 1
shortcut = True
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 1024
ch_out = 256
stride = 1
shortcut = True
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 1024
ch_out = 256
stride = 1
shortcut = True
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 1024
ch_out = 256
stride = 1
shortcut = True
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 1024
ch_out = 512
stride = 2
shortcut = False
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 2048
ch_out = 512
stride = 1
shortcut = True
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 2048
ch_out = 512
stride = 1
shortcut = True
variant = 'd'
groups = 1
base_width = 64
lr = 0.5
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False





torch.backends.cudnn.benchmark = True  # Improves training speed.
torch.backends.cuda.matmul.allow_tf32 = False  # Allow PyTorch to internally use tf32 for matmul
torch.backends.cudnn.allow_tf32 = False  # Allow PyTorch to internally use tf32 for convolutions

model = BottleNeck(ch_in=ch_in,
                                ch_out=ch_out,
                                stride=stride,
                                shortcut=shortcut,
                                variant=variant,
                                groups=groups,
                                base_width=base_width,
                                lr=lr,
                                norm_type=norm_type,
                                norm_decay=norm_decay,
                                freeze_norm=freeze_norm,
                                dcn_v2=dcn_v2,
                                std_senet=std_senet,
                                )
model.train()
need_clip = False

base_lr = 0.00000001 * 1.0
param_groups = []
base_wd = 0.0005
# base_wd = 0.0
momentum = 0.9
# 是否进行梯度裁剪
need_clip = False
clip_norm = 1000000.0
# need_clip = True
# clip_norm = 35.0
model.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
optimizer = torch.optim.SGD(param_groups, lr=base_lr, momentum=momentum, weight_decay=base_wd)
model.load_state_dict(torch.load("53_00.pth", map_location=torch.device('cpu')))
model.fix_bn()

use_gpu = True
if use_gpu:
    model = model.cuda()

dic2 = np.load('53.npz')
print(torch.__version__)
for batch_idx in range(8):
    print('======================== batch_%.3d ========================'%batch_idx)
    optimizer.zero_grad(set_to_none=True)
    x = dic2['batch_%.3d.x'%batch_idx]
    y_paddle = dic2['batch_%.3d.y'%batch_idx]
    branch2a_conv_w_grad_paddle = dic2['batch_%.3d.branch2a_conv_w_grad'%batch_idx]
    if not freeze_norm:
        branch2a_norm_w_grad_paddle = dic2['batch_%.3d.branch2a_norm_w_grad'%batch_idx]
        branch2a_norm_b_grad_paddle = dic2['batch_%.3d.branch2a_norm_b_grad'%batch_idx]

    branch2b_conv_w_grad_paddle = dic2['batch_%.3d.branch2b_conv_w_grad'%batch_idx]
    if not freeze_norm:
        branch2b_norm_w_grad_paddle = dic2['batch_%.3d.branch2b_norm_w_grad'%batch_idx]
        branch2b_norm_b_grad_paddle = dic2['batch_%.3d.branch2b_norm_b_grad'%batch_idx]

    branch2c_conv_w_grad_paddle = dic2['batch_%.3d.branch2c_conv_w_grad'%batch_idx]
    if not freeze_norm:
        branch2c_norm_w_grad_paddle = dic2['batch_%.3d.branch2c_norm_w_grad'%batch_idx]
        branch2c_norm_b_grad_paddle = dic2['batch_%.3d.branch2c_norm_b_grad'%batch_idx]

    x = torch.Tensor(x)
    if use_gpu:
        x = x.cuda()
    x.requires_grad_(True)

    y = model(x)

    y_pytorch = y.cpu().detach().numpy()
    ddd = np.sum((y_pytorch - y_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    loss = y.sum()
    loss.backward()

    # 注意，这是裁剪之前的梯度，Paddle无法获得裁剪后的梯度。
    branch2a_conv_w_grad_pytorch = model.branch2a.conv.weight.grad.cpu().detach().numpy()
    ddd = np.mean((branch2a_conv_w_grad_pytorch - branch2a_conv_w_grad_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    if not freeze_norm:
        branch2a_norm_w_grad_pytorch = model.branch2a.norm.weight.grad.cpu().detach().numpy()
        ddd = np.mean((branch2a_norm_w_grad_pytorch - branch2a_norm_w_grad_paddle) ** 2)
        print('ddd=%.6f' % ddd)

        branch2a_norm_b_grad_pytorch = model.branch2a.norm.bias.grad.cpu().detach().numpy()
        ddd = np.mean((branch2a_norm_b_grad_pytorch - branch2a_norm_b_grad_paddle) ** 2)
        print('ddd=%.6f' % ddd)



    branch2b_conv_w_grad_pytorch = model.branch2b.conv.weight.grad.cpu().detach().numpy()
    ddd = np.mean((branch2b_conv_w_grad_pytorch - branch2b_conv_w_grad_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    if not freeze_norm:
        branch2b_norm_w_grad_pytorch = model.branch2b.norm.weight.grad.cpu().detach().numpy()
        ddd = np.mean((branch2b_norm_w_grad_pytorch - branch2b_norm_w_grad_paddle) ** 2)
        print('ddd=%.6f' % ddd)

        branch2b_norm_b_grad_pytorch = model.branch2b.norm.bias.grad.cpu().detach().numpy()
        ddd = np.mean((branch2b_norm_b_grad_pytorch - branch2b_norm_b_grad_paddle) ** 2)
        print('ddd=%.6f' % ddd)



    branch2c_conv_w_grad_pytorch = model.branch2c.conv.weight.grad.cpu().detach().numpy()
    ddd = np.mean((branch2c_conv_w_grad_pytorch - branch2c_conv_w_grad_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    if not freeze_norm:
        branch2c_norm_w_grad_pytorch = model.branch2c.norm.weight.grad.cpu().detach().numpy()
        ddd = np.mean((branch2c_norm_w_grad_pytorch - branch2c_norm_w_grad_paddle) ** 2)
        print('ddd=%.6f' % ddd)

        branch2c_norm_b_grad_pytorch = model.branch2c.norm.bias.grad.cpu().detach().numpy()
        ddd = np.mean((branch2c_norm_b_grad_pytorch - branch2c_norm_b_grad_paddle) ** 2)
        print('ddd=%.6f' % ddd)

    # 梯度裁剪
    if need_clip:
        for param_group in optimizer.param_groups:
            if param_group['need_clip']:
                torch.nn.utils.clip_grad_norm_(param_group['params'], max_norm=param_group['clip_norm'], norm_type=2)
    optimizer.step()
torch.save(model.state_dict(), "53_08.pth")
print(torch.__version__)
print()
