import numpy as np
import sys
import torch
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from test_code.ppdet_resnet import ResNet, load_weight


def build_2d_sincos_position_embedding(w,
                                       h,
                                       embed_dim=256,
                                       temperature=10000.):
    grid_w = paddle.arange(int(w), dtype=paddle.float32)
    grid_h = paddle.arange(int(h), dtype=paddle.float32)
    grid_w, grid_h = paddle.meshgrid(grid_w, grid_h)
    assert embed_dim % 4 == 0, \
        'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = embed_dim // 4
    omega = paddle.arange(pos_dim, dtype=paddle.float32) / pos_dim
    omega = 1. / (temperature ** omega)

    out_w = grid_w.flatten()[..., None] @ omega[None]
    out_h = grid_h.flatten()[..., None] @ omega[None]

    aaaaaaaa = paddle.concat(
        [
            paddle.sin(out_w), paddle.cos(out_w), paddle.sin(out_h),
            paddle.cos(out_h)
        ],
        axis=1)
    aa = aaaaaaaa[None, :, :]
    return aa




def build_2d_sincos_position_embedding2(w,
                                       h,
                                       embed_dim=256,
                                       temperature=10000.):
    grid_w = torch.arange(int(w), dtype=torch.float32)
    grid_h = torch.arange(int(h), dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim   # [pos_dim, ]  下标归一化到0到1之间
    omega = 1. / (temperature**omega)    # [pos_dim, ]  omega从1递减到接近 1/temperature
    omega = omega.unsqueeze(0)   # [1, pos_dim]

    grid_w = torch.reshape(grid_w, (h*w, 1))   # [h*w, 1]
    grid_h = torch.reshape(grid_h, (h*w, 1))   # [h*w, 1]

    out_w = grid_w @ omega   # 矩阵乘, [h*w, pos_dim]
    out_h = grid_h @ omega   # 矩阵乘, [h*w, pos_dim]

    out = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)  # [h*w, 4*pos_dim]
    out = out.unsqueeze(0)   # [1, h*w, hidden_dim]
    return out


seed = 13
img = np.random.RandomState(seed).randn(2, 256, 20, 20)
inputs = paddle.to_tensor(img, dtype=paddle.float32)

w = 20
h = 20
hidden_dim = 256
pe_temperature = 10000.
pos_embed = build_2d_sincos_position_embedding(w, h, hidden_dim, pe_temperature)
pos_embed2 = build_2d_sincos_position_embedding2(w, h, hidden_dim, pe_temperature)


v00 = pos_embed.numpy()
v01 = pos_embed2.cpu().detach().numpy()
ddd = np.sum((v00 - v01) ** 2)
print('ddd val=%.6f' % ddd)

print('Done.')




