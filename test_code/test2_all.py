

import numpy as np
import torch
import paddle
import torch.nn.functional as F
import paddle.nn.functional as F2

'''
dropout:
对于张量每个元素，有p的概率被置为0，有(1-p)的概率被乘以1/(1-p)倍;
当training=False, 就变成原值输出。
'''

anchors = np.ones([2, 2, 2]) * 0.5
anchors = anchors.astype(np.float32)
anchors[0, 0, 0] = 0.0001
anchors[0, 0, 1] = 0.9999
anchors[0, 1, 0] = 0.2
anchors[0, 1, 1] = 0.3
print(anchors)
eps = 0.01


anchors2 = paddle.to_tensor(anchors)
valid_mask2 = ((anchors2 > eps) & (anchors2 < 1 - eps)).all(-1, keepdim=True)

anchors9 = paddle.log(anchors2 / (1 - anchors2))
anchors8 = paddle.where(valid_mask2, anchors9, paddle.to_tensor(float("inf")))


anchors7 = paddle.where(valid_mask2, anchors9, paddle.ones_like(anchors9) * 100000.)

anchors6 = paddle.where(valid_mask2, anchors9, paddle.to_tensor(0.))



anchors3 = torch.Tensor(anchors)
valid_mask = ((anchors3 > eps) & (anchors3 < 1 - eps)).all(-1, keepdim=True)


anchors3 = torch.log(anchors3 / (1 - anchors3))
anchors3 = torch.where(valid_mask, anchors3, paddle.to_tensor(float("inf")))



print()



