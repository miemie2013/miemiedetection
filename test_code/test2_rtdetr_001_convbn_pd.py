import numpy as np
import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.regularizer import L2Decay


class MyNet(nn.Layer):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv = nn.Conv2D(3, 2, kernel_size=3, stride=2, padding=1, bias_attr=False)
        freeze_norm = False
        norm_lr = 1.0
        norm_decay = 0.0
        param_attr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay),
            trainable=False if freeze_norm else True)
        bias_attr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay),
            trainable=False if freeze_norm else True)

        global_stats = True if freeze_norm else None
        self.bn = nn.BatchNorm2D(
            2,
            weight_attr=param_attr,
            bias_attr=bias_attr,
            use_global_stats=global_stats)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


model = MyNet()

with paddle.no_grad():
    weight_ = np.random.RandomState(113).randn(2, 3, 3, 3).astype(np.float32)
    model.conv.weight.set_value(paddle.to_tensor(weight_))
    bn_w = np.random.RandomState(114).randn(2).astype(np.float32)
    bn_b = np.random.RandomState(115).randn(2).astype(np.float32)
    model.bn.weight.set_value(paddle.to_tensor(bn_w))
    model.bn.bias.set_value(paddle.to_tensor(bn_b))

grad_clip = None
# grad_clip = nn.ClipGradByGlobalNorm(clip_norm=0.1)
'''
momentum = 0.9
lr = 0.9
weight_decay = 0.9999
_params = model.parameters()
params = [param for param in _params if param.trainable is True]
optimizer = paddle.optimizer.Momentum(learning_rate=lr, parameters=params,
                                      grad_clip=grad_clip, momentum=momentum,
                                      weight_decay=paddle.regularizer.L2Decay(weight_decay))
'''

lr = 0.9
weight_decay = 0.0001
_params = model.parameters()
params = [param for param in _params if param.trainable is True]
optimizer = paddle.optimizer.AdamW(learning_rate=lr, parameters=params, grad_clip=grad_clip, weight_decay=weight_decay)

seed = 12
model.train()
for step_id in range(1):
    print('======================= step=%d ======================='%step_id)
    model.train()
    img = np.random.RandomState(seed).randn(2, 3, 4, 4).astype(np.float32)
    inputs = paddle.to_tensor(img, dtype=paddle.float32)
    outs = model(inputs)
    print(outs)
    loss = outs.mean()

    loss.backward()
    optimizer.step()
    optimizer.clear_grad()
    seed += 10

model.eval()
img = np.random.RandomState(seed+1).randn(2, 3, 4, 4).astype(np.float32)
inputs = paddle.to_tensor(img, dtype=paddle.float32)
outs = model(inputs)

print('======================= eval =======================')
print(outs)
dic = {}
dic['outs'] = outs.numpy()
np.savez('001', **dic)


print('Done.')




