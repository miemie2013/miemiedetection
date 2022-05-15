import paddle
import numpy as np
import test_grad.ppdet_resnet as ppdet_resnet
from paddle.regularizer import L2Decay


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



x_shape = [1, ch_in, 32, 32]


batch_size = 4
fused_modconv = False

model = ppdet_resnet.BottleNeck(ch_in=ch_in,
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
base_lr = 0.00000001 * 1.0
base_wd = 0.0005
# base_wd = 0.0
momentum = 0.9
# 是否进行梯度裁剪
clip_grad_by_norm = None
# clip_grad_by_norm = 35.0
if clip_grad_by_norm is not None:
    grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=clip_grad_by_norm)
else:
    grad_clip = None

weight_decay = L2Decay(base_wd) if base_wd > 0.0000000001 else None
optimizer = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=base_lr,
                                      momentum=momentum, weight_decay=weight_decay, grad_clip=grad_clip)
paddle.save(model.state_dict(), "53_00.pdparams")

dic = {}
for batch_idx in range(8):
    optimizer.clear_gradients()

    x_shape[0] = batch_size
    x = paddle.randn(x_shape)
    x.stop_gradient = False

    y = model(x)

    dic['batch_%.3d.y'%batch_idx] = y.numpy()
    dic['batch_%.3d.x'%batch_idx] = x.numpy()

    loss = y.sum()
    loss.backward()

    # 注意，这是裁剪之前的梯度，Paddle无法获得裁剪后的梯度。
    dic['batch_%.3d.branch2a_conv_w_grad'%batch_idx] = model.branch2a.conv.weight.grad.numpy()
    if not freeze_norm:
        dic['batch_%.3d.branch2a_norm_w_grad'%batch_idx] = model.branch2a.norm.weight.grad.numpy()
        dic['batch_%.3d.branch2a_norm_b_grad'%batch_idx] = model.branch2a.norm.bias.grad.numpy()

    dic['batch_%.3d.branch2b_conv_w_grad'%batch_idx] = model.branch2b.conv.weight.grad.numpy()
    if not freeze_norm:
        dic['batch_%.3d.branch2b_norm_w_grad'%batch_idx] = model.branch2b.norm.weight.grad.numpy()
        dic['batch_%.3d.branch2b_norm_b_grad'%batch_idx] = model.branch2b.norm.bias.grad.numpy()

    dic['batch_%.3d.branch2c_conv_w_grad'%batch_idx] = model.branch2c.conv.weight.grad.numpy()
    if not freeze_norm:
        dic['batch_%.3d.branch2c_norm_w_grad'%batch_idx] = model.branch2c.norm.weight.grad.numpy()
        dic['batch_%.3d.branch2c_norm_b_grad'%batch_idx] = model.branch2c.norm.bias.grad.numpy()
    optimizer.step()
np.savez('53', **dic)
paddle.save(model.state_dict(), "53_08.pdparams")
print(paddle.__version__)
print()
