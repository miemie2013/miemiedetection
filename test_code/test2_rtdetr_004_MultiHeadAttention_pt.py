import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.initializer import xavier_uniform_, constant_


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



def _convert_attention_mask(attn_mask, dtype):
    """
    Convert the attention mask to the target dtype we expect.
    Parameters:
        attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False`
                values and the others have `True` values. When the data type is
                int, the unwanted positions have 0 values and the others have 1
                values. When the data type is float, the unwanted positions have
                `-INF` values and the others have 0 values. It can be None when
                nothing wanted or needed to be prevented attention to. Default None.
        dtype (VarType): The target type of `attn_mask` we expect.
    Returns:
        Tensor: A Tensor with shape same as input `attn_mask`, with data type `dtype`.
    """
    return -1

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 need_weights=False):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim:
            # 要注意，pytorch的fc层和paddle的fc层的权重weight需要转置一下才能等价！！！
            self.in_proj_weight = torch.nn.Parameter(torch.randn([3 * embed_dim, embed_dim]))
            self.in_proj_bias = torch.nn.Parameter(torch.full([3 * embed_dim], np.float32(0.)))
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(self.kdim, embed_dim)
            self.v_proj = nn.Linear(self.vdim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self._type_list = ('q_proj', 'k_proj', 'v_proj')

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
            else:
                constant_(p)

    def compute_qkv(self, tensor, index):
        if self._qkv_same_embed_dim:
            # 要注意，pytorch的fc层和paddle的fc层的权重weight需要转置一下才能等价！！！
            weight = self.in_proj_weight[index * self.embed_dim:(index + 1) * self.embed_dim, :]
            bias = self.in_proj_bias[index * self.embed_dim:(index + 1) * self.embed_dim] if self.in_proj_bias is not None else None
            tensor = F.linear(tensor, weight=weight, bias=bias)
        else:
            tensor = getattr(self, self._type_list[index])(tensor)
        N, HW, _ = tensor.shape
        tensor = tensor.reshape([N, HW, self.num_heads, self.head_dim])
        tensor = tensor.permute([0, 2, 1, 3])   # [N, num_heads, HW, head_dim]
        return tensor

    def forward(self, query, key=None, value=None, attn_mask=None):
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        q, k, v = (self.compute_qkv(t, i)
                   for i, t in enumerate([query, key, value]))

        # scale dot product attention
        product = q.matmul(k.permute([0, 1, 3, 2]))    # [N, num_heads, HW, HW]
        scaling = float(self.head_dim)**-0.5
        product = product * scaling

        if attn_mask is not None:
            # Support bool or int mask
            attn_mask = _convert_attention_mask(attn_mask, product.dtype)
            product = product + attn_mask
        # paddle的softmax dim默认是-1，所以这里显式写上-1
        weights = F.softmax(product, dim=-1)
        if self.dropout:
            weights = F.dropout(
                weights,
                self.dropout,
                training=self.training)
        out = torch.matmul(weights, v)    # [N, num_heads, HW, head_dim]

        # combine heads
        out = out.permute([0, 2, 1, 3])    # [N, HW, num_heads, head_dim]
        N, HW, _, _ = out.shape
        out = torch.reshape(out, [N, HW, out.shape[2] * out.shape[3]])    # [N, HW, embed_dim]

        # project to output
        out = self.out_proj(out)    # [N, HW, embed_dim]

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        return out if len(outs) == 1 else tuple(outs)


d_model = 256
nhead = 8
attn_dropout = 0.0
model = MultiHeadAttention(d_model, nhead, attn_dropout)


test_idx = '004'
state_dict = model.state_dict()

dic_weight = np.load('%s_w.npz'%test_idx)
for param_name in state_dict.keys():
    vvvvvvv = dic_weight[param_name]
    vvvvvvv2 = torch.Tensor(vvvvvvv)
    if param_name.endswith('out_proj.weight'):
        vvvvvvv2 = vvvvvvv2.permute((1, 0))
    if param_name.endswith('in_proj_weight'):
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


def with_pos_embed(tensor, pos_embed):
    return tensor if pos_embed is None else tensor + pos_embed

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
    q = k = with_pos_embed(src, pos_embed)
    yyy = model(q, k, value=src, attn_mask=src_mask)
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
q = k = with_pos_embed(src, pos_embed)
yyy = model(q, k, value=src, attn_mask=src_mask)


dic2 = np.load('%s.npz'%test_idx)
yyy2 = dic2['yyy']

yyy3 = yyy.cpu().detach().numpy()
ddd = np.mean((yyy3 - yyy2) ** 2)
print('ddd val=%.6f' % ddd)
print('Done.')




