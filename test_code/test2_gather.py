

import numpy as np
import torch
import paddle
from mmdet.models.ops import gather_1d

'''

'''

x = np.random.normal(size=[5, 2])
# x = np.random.normal(size=[5, ])
index = np.array([0, 1, 2, 3]).astype(np.int32)

x2 = paddle.to_tensor(x)
index2 = paddle.to_tensor(index)


y2 = paddle.gather(x2, index2, axis=0)


x = torch.Tensor(x)
index = torch.Tensor(index).to(torch.int64)
y = gather_1d(x, index)


# ddd = np.sum((temp32.cpu().detach().numpy() - temp3.numpy())**2)
# print('ddd=%.6f' % ddd)
#
# ddd = np.sum((w_avg2.cpu().detach().numpy() - temp3.numpy())**2)
# print('ddd=%.6f' % ddd)

print()



