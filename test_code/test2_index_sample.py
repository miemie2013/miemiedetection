

import numpy as np
import torch
import paddle
from mmdet.models.ops import index_sample_2d

'''

'''

x = np.array([[1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0]]).astype(np.float32)
index = np.array([[0, 1, 2],
                        [1, 2, 3],
                        [0, 0, 0]]).astype(np.int32)

x2 = paddle.to_tensor(x)
index2 = paddle.to_tensor(index)
y2 = paddle.index_sample(x2, index2)

x = torch.Tensor(x)
index = torch.Tensor(index).to(torch.int64)
y = index_sample_2d(x, index)


# ddd = np.sum((temp32.cpu().detach().numpy() - temp3.numpy())**2)
# print('ddd=%.6f' % ddd)
#
# ddd = np.sum((w_avg2.cpu().detach().numpy() - temp3.numpy())**2)
# print('ddd=%.6f' % ddd)

print()



