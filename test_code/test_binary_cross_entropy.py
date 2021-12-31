import paddle
import numpy as np
import paddle.nn.functional as F

'''
YOLOv3Loss 中使用过。
'''

y = np.array([[0.12, 0.93], [0.55, 0.21]]).astype(np.float32)
ty = np.array([[0.6, 0.65], [0.34, 0.75]]).astype(np.float32)

y222 = paddle.to_tensor(y)
ty222 = paddle.to_tensor(ty)
loss_y222 = F.binary_cross_entropy(y222, ty222, reduction='none')


import torch
y = torch.Tensor(y).cuda()
ty = torch.Tensor(ty).cuda()

def binary_cross_entropy(p, labels, eps=1e-9):
    pos_loss = labels * (0 - torch.log(p + eps))
    neg_loss = (1.0 - labels) * (0 - torch.log(1 - p + eps))
    bce_loss = pos_loss + neg_loss
    return bce_loss


loss_y = binary_cross_entropy(y, ty)

# 结果和pytorch自带的F.binary_cross_entropy()一样。
import torch.nn.functional as F
loss_y = F.binary_cross_entropy(y, ty, reduction='none')


print()









