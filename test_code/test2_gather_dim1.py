

import numpy as np
import torch
import paddle
from mmdet.models.ops import gather_1d_dim1

'''

'''

x = np.random.normal(size=[5, 4])
# x = np.random.normal(size=[5, ])
index = np.array([0, 1, 2]).astype(np.int32)

x2 = paddle.to_tensor(x)
index2 = paddle.to_tensor(index)

aaaaaaa1 = x2.chunk(5)
y2 = paddle.gather(x2, index2, axis=1)


x = torch.Tensor(x)
index = torch.Tensor(index).to(torch.int64)
aaaaaaa2 = x.chunk(5)
y = gather_1d_dim1(x, index)


ddd = np.sum((y.cpu().detach().numpy() - y2.numpy())**2)
print('ddd=%.6f' % ddd)


print()



