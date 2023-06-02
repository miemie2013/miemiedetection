import numpy as np
import sys
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from test_code.ppdet_resnet import ResNet, load_weight


class MyNet(nn.Layer):
    def __init__(self, backbone):
        super(MyNet, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        outs = self.backbone(x)
        return outs

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




