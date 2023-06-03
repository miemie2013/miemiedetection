import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.initializer import linear_init_
from mmdet.models.layers import MultiHeadAttention
from test_code.ppdet_resnet import ResNet, load_weight


def build_2d_sincos_position_embedding(w, h, device,
                                       embed_dim=256,
                                       temperature=10000.):
    grid_w = torch.arange(int(w), dtype=torch.float32, device=device)
    grid_h = torch.arange(int(h), dtype=torch.float32, device=device)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim   # [pos_dim, ]  下标归一化到0到1之间
    omega = 1. / (temperature**omega)    # [pos_dim, ]  omega从1递减到接近 1/temperature
    omega = omega.unsqueeze(0)   # [1, pos_dim]

    grid_w = torch.reshape(grid_w, (h*w, 1))   # [h*w, 1]
    grid_h = torch.reshape(grid_h, (h*w, 1))   # [h*w, 1]

    out_w = grid_w @ omega   # 矩阵乘, [h*w, pos_dim]
    out_h = grid_h @ omega   # 矩阵乘, [h*w, pos_dim]

    out = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)  # [h*w, 4*pos_dim]
    out = out.unsqueeze(0)   # [1, h*w, hidden_dim]
    return out


class TransformerLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False):
        super(TransformerLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.self_attn = MultiHeadAttention(d_model, nhead, attn_dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(act_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = getattr(F, activation)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


hidden_dim = 256
encoder_layer_ = dict(
    d_model=hidden_dim,
    nhead=8,
    dim_feedforward=1024,
    dropout=0.,
    activation='gelu',
)

model = TransformerLayer(**encoder_layer_)

test_idx = '005'
state_dict = model.state_dict()

dic_weight = np.load('%s_w.npz'%test_idx)
for param_name in state_dict.keys():
    vvvvvvv = dic_weight[param_name]
    vvvvvvv2 = torch.Tensor(vvvvvvv)
    if param_name.endswith('out_proj.weight'):
        vvvvvvv2 = vvvvvvv2.permute((1, 0))
    if param_name.endswith('in_proj_weight'):
        vvvvvvv2 = vvvvvvv2.permute((1, 0))
    if param_name.endswith('linear1.weight'):
        vvvvvvv2 = vvvvvvv2.permute((1, 0))
    if param_name.endswith('linear2.weight'):
        vvvvvvv2 = vvvvvvv2.permute((1, 0))
    state_dict[param_name] = vvvvvvv2
model.load_state_dict(state_dict)


momentum = 0.9
lr = 0.005
weight_decay = 0.0005

# 不可以加正则化的参数：norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数。
no_L2, use_L2 = [], []
for name, param in model.named_parameters():
    use_L2.append(param)
optimizer = torch.optim.SGD(use_L2, lr=lr, momentum=momentum, weight_decay=weight_decay)


w = 20
h = 20
hidden_dim = 256
pe_temperature = 10000.
seed = 12
model.cuda()
model.train()
device = torch.ones((7, )).to(torch.float32).cuda().device
for step_id in range(8):
    print('======================= step=%d ======================='%step_id)
    model.train()
    img = np.random.RandomState(seed).randn(2, 20*20, 256)
    src = torch.Tensor(img).to(torch.float32).cuda()

    pos_embed = build_2d_sincos_position_embedding(w, h, device, hidden_dim, pe_temperature)
    src_mask = None
    yyy = model(src, src_mask=src_mask, pos_embed=pos_embed)
    loss = yyy.mean()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    seed += 10

model.eval()
img = np.random.RandomState(seed+1).randn(2, 20*20, 256)
src = torch.Tensor(img).to(torch.float32).cuda()
pos_embed = build_2d_sincos_position_embedding(w, h, device, hidden_dim, pe_temperature)
src_mask = None
yyy = model(src, src_mask=src_mask, pos_embed=pos_embed)


dic2 = np.load('%s.npz'%test_idx)
yyy2 = dic2['yyy']

yyy3 = yyy.cpu().detach().numpy()
ddd = np.mean((yyy3 - yyy2) ** 2)
print('ddd val=%.6f' % ddd)
print('Done.')




