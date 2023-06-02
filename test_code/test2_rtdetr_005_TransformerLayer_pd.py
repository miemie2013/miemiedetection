import numpy as np
import sys
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from test_code.ppdet_resnet import ResNet, load_weight
from test_code.ppdet_init import xavier_uniform_, constant_, linear_init_


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
    return nn.layer.transformer._convert_attention_mask(attn_mask, dtype)

class MultiHeadAttention(nn.Layer):
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.

    Please refer to `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_
    for more details.

    Parameters:
        embed_dim (int): The expected feature size in the input and output.
        num_heads (int): The number of heads in multi-head attention.
        dropout (float, optional): The dropout probability used on attention
            weights to drop some attention targets. 0 for no dropout. Default 0
        kdim (int, optional): The feature size in key. If None, assumed equal to
            `embed_dim`. Default None.
        vdim (int, optional): The feature size in value. If None, assumed equal to
            `embed_dim`. Default None.
        need_weights (bool, optional): Indicate whether to return the attention
            weights. Default False.

    Examples:

        .. code-block:: python

            import paddle

            # encoder input: [batch_size, sequence_length, d_model]
            query = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, num_heads, query_len, query_len]
            attn_mask = paddle.rand((2, 2, 4, 4))
            multi_head_attn = paddle.nn.MultiHeadAttention(128, 2)
            output = multi_head_attn(query, None, None, attn_mask=attn_mask)  # [2, 4, 128]
    """

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
            self.in_proj_weight = self.create_parameter(
                shape=[embed_dim, 3 * embed_dim],
                attr=None,
                dtype=self._dtype,
                is_bias=False)
            self.in_proj_bias = self.create_parameter(
                shape=[3 * embed_dim],
                attr=None,
                dtype=self._dtype,
                is_bias=True)
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
            weight = self.in_proj_weight[:, index * self.embed_dim:(index + 1) * self.embed_dim]
            bias = self.in_proj_bias[index * self.embed_dim:(index + 1) * self.embed_dim] if self.in_proj_bias is not None else None
            tensor = F.linear(x=tensor, weight=weight, bias=bias)
        else:
            tensor = getattr(self, self._type_list[index])(tensor)
        N, HW, _ = tensor.shape
        tensor = tensor.reshape([N, HW, self.num_heads, self.head_dim])
        tensor = tensor.transpose([0, 2, 1, 3])   # [N, num_heads, HW, head_dim]
        return tensor

    def forward(self, query, key=None, value=None, attn_mask=None):
        r"""
        Applies multi-head attention to map queries and a set of key-value pairs
        to outputs.

        Parameters:
            query (Tensor): The queries for multi-head attention. It is a
                tensor with shape `[batch_size, query_length, embed_dim]`. The
                data type should be float32 or float64.
            key (Tensor, optional): The keys for multi-head attention. It is
                a tensor with shape `[batch_size, key_length, kdim]`. The
                data type should be float32 or float64. If None, use `query` as
                `key`. Default None.
            value (Tensor, optional): The values for multi-head attention. It
                is a tensor with shape `[batch_size, value_length, vdim]`.
                The data type should be float32 or float64. If None, use `query` as
                `value`. Default None.
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

        Returns:
            Tensor|tuple: It is a tensor that has the same shape and data type \
                as `query`, representing attention output. Or a tuple if \
                `need_weights` is True or `cache` is not None. If `need_weights` \
                is True, except for attention output, the tuple also includes \
                the attention weights tensor shaped `[batch_size, num_heads, query_length, key_length]`. \
                If `cache` is not None, the tuple then includes the new cache \
                having the same type as `cache`, and if it is `StaticCache`, it \
                is same as the input `cache`, if it is `Cache`, the new cache \
                reserves tensors concatanating raw tensors with intermediate \
                results of current query.
        """
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        q, k, v = (self.compute_qkv(t, i)
                   for i, t in enumerate([query, key, value]))

        # scale dot product attention
        product = paddle.matmul(x=q, y=k, transpose_y=True)    # [N, num_heads, HW, HW]
        scaling = float(self.head_dim)**-0.5
        product = product * scaling

        if attn_mask is not None:
            # Support bool or int mask
            attn_mask = _convert_attention_mask(attn_mask, product.dtype)
            product = product + attn_mask
        weights = F.softmax(product)
        if self.dropout:
            weights = F.dropout(
                weights,
                self.dropout,
                training=self.training,
                mode="upscale_in_train")
        out = paddle.matmul(weights, v)    # [N, num_heads, HW, head_dim]

        # combine heads
        out = paddle.transpose(out, perm=[0, 2, 1, 3])    # [N, HW, num_heads, head_dim]
        N, HW, _, _ = out.shape
        out = paddle.reshape(x=out, shape=[N, HW, out.shape[2] * out.shape[3]])    # [N, HW, embed_dim]

        # project to output
        out = self.out_proj(out)    # [N, HW, embed_dim]

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        return out if len(outs) == 1 else tuple(outs)


class TransformerLayer(nn.Layer):
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
        self.dropout = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(dropout, mode="upscale_in_train")
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

dic_weight = {}
for param_name in state_dict.keys():
    param_ = state_dict[param_name]
    dic_weight[param_name] = param_.numpy()
np.savez('%s_w'%test_idx, **dic_weight)


grad_clip = None
momentum = 0.9
lr = 0.005
weight_decay = 0.0005
_params = model.parameters()
params = [param for param in _params if param.trainable is True]
optimizer = paddle.optimizer.Momentum(learning_rate=lr, parameters=params,
                                      grad_clip=grad_clip, momentum=momentum,
                                      weight_decay=paddle.regularizer.L2Decay(weight_decay))


w = 20
h = 20
hidden_dim = 256
pe_temperature = 10000.
seed = 12
model.train()
for step_id in range(8):
    print('======================= step=%d ======================='%step_id)
    model.train()
    img = np.random.RandomState(seed).randn(2, 20*20, 256)
    src = paddle.to_tensor(img, dtype=paddle.float32)

    pos_embed = build_2d_sincos_position_embedding(w, h, hidden_dim, pe_temperature)
    src_mask = None
    yyy = model(src, src_mask=src_mask, pos_embed=pos_embed)
    loss = yyy.mean()

    loss.backward()
    optimizer.step()
    optimizer.clear_grad()
    seed += 10

model.eval()
img = np.random.RandomState(seed+1).randn(2, 20*20, 256)
src = paddle.to_tensor(img, dtype=paddle.float32)
pos_embed = build_2d_sincos_position_embedding(w, h, hidden_dim, pe_temperature)
src_mask = None
yyy = model(src, src_mask=src_mask, pos_embed=pos_embed)


dic = {}
dic['yyy'] = yyy.numpy()
np.savez(test_idx, **dic)




print('Done.')




