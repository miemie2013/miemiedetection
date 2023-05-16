

import numpy as np
import torch
import paddle
import torch.nn.functional as F
import paddle.nn.functional as F2

'''

'''

pred_corners = np.random.normal(size=[124, 17])
target_corners = np.random.normal(size=[124, 17])



pred_corners2 = paddle.to_tensor(pred_corners)
target_corners2 = paddle.to_tensor(target_corners)

target_corners_label2 = F2.softmax(target_corners2, -1)
loss_dfl2 = F2.cross_entropy(
    pred_corners2,
    target_corners_label2,
    soft_label=True,
    reduction='none')
loss_dfl22 = loss_dfl2.sum(1)


pred_corners = torch.Tensor(pred_corners)
target_corners = torch.Tensor(target_corners)

target_corners_label = F.softmax(target_corners, -1)


a = 8
loss_dfl = F.cross_entropy(
    pred_corners,
    target_corners_label,
    reduction='none')
# loss_dfl333 = loss_dfl.sum(1)


ddd = np.sum((loss_dfl.cpu().detach().numpy() - loss_dfl22.numpy())**2)
print('ddd=%.6f' % ddd)




ddd = np.sum((w_avg2.cpu().detach().numpy() - temp3.numpy())**2)
print('ddd=%.6f' % ddd)

print()



