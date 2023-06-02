import numpy as np
import sys
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from test_code.ppdet_resnet import ResNet, load_weight


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

param_path = '../ppyolov2_r50vd_dcn_365e_coco.pdparams'
load_weight(model, param_path)


grad_clip = None
# grad_clip = nn.ClipGradByGlobalNorm(clip_norm=0.1)
momentum = 0.9
lr = 0.005
weight_decay = 0.0005
_params = model.parameters()
params = [param for param in _params if param.trainable is True]
optimizer = paddle.optimizer.Momentum(learning_rate=lr, parameters=params,
                                      grad_clip=grad_clip, momentum=momentum,
                                      weight_decay=paddle.regularizer.L2Decay(weight_decay))

seed = 12
model.train()
for step_id in range(8):
    print('======================= step=%d ======================='%step_id)
    model.train()
    img = np.random.RandomState(seed).randn(2, 3, 640, 640)
    aaa = paddle.to_tensor(img, dtype=paddle.float32)
    inputs = {'image': aaa}
    outs = model(inputs)
    loss = outs[0].mean() + outs[1].mean() + outs[2].mean()

    loss.backward()
    optimizer.step()
    optimizer.clear_grad()
    seed += 10

model.eval()
img = np.random.RandomState(seed+1).randn(2, 3, 640, 640)
aaa = paddle.to_tensor(img, dtype=paddle.float32)
inputs = {'image': aaa}
outs = model(inputs)

dic = {}
dic['outs0'] = outs[0].numpy()
dic['outs1'] = outs[1].numpy()
dic['outs2'] = outs[2].numpy()
np.savez('002', **dic)


print('Done.')




