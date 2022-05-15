import paddle
import numpy as np
import test_grad.ppdet_resnet as ppdet_resnet
from paddle.regularizer import L2Decay


depth = 50
variant = 'd'
return_idx = [1, 2, 3]
dcn_v2_stages = [-1]
freeze_at = -1
freeze_norm = False
norm_decay = 0.

depth = 50
variant = 'd'
return_idx = [1, 2, 3]
dcn_v2_stages = [-1]
freeze_at = 2
freeze_norm = False
norm_decay = 0.



x_shape = [1, 3, 416, 416]


batch_size = 4
fused_modconv = False

model = ppdet_resnet.ResNet(depth=depth, variant=variant, return_idx=return_idx, dcn_v2_stages=dcn_v2_stages,
                            freeze_at=freeze_at, freeze_norm=freeze_norm, norm_decay=norm_decay)
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
paddle.save(model.state_dict(), "51_00.pdparams")

dic = {}
for batch_idx in range(8):
    optimizer.clear_gradients()

    x_shape[0] = batch_size
    x = paddle.randn(x_shape)
    x.stop_gradient = False

    y = model(dict(image=x,))
    y_52x52 = y[0]
    y_26x26 = y[1]
    y_13x13 = y[2]

    dic['batch_%.3d.y_52x52'%batch_idx] = y_52x52.numpy()
    dic['batch_%.3d.y_26x26'%batch_idx] = y_26x26.numpy()
    dic['batch_%.3d.y_13x13'%batch_idx] = y_13x13.numpy()
    dic['batch_%.3d.x'%batch_idx] = x.numpy()

    loss = y_13x13.sum()
    loss.backward()

    # 注意，这是裁剪之前的梯度，Paddle无法获得裁剪后的梯度。
    # dic['batch_%.3d.w_grad'%batch_idx] = model.res5.res5c.branch2c.conv.weight.grad.numpy()
    # dic['batch_%.3d.b_grad'%batch_idx] = model.res5.res5c.branch2c.norm.bias.grad.numpy()
    optimizer.step()
np.savez('51', **dic)
paddle.save(model.state_dict(), "51_08.pdparams")
print(paddle.__version__)
print()
