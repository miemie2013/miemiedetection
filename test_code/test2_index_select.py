

import numpy as np
import torch
import paddle
from mmdet.models.ops import index_sample_2d

'''

'''

num_classes = 3
x = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]).astype(np.float32)
index = np.array([0, 1, 2]).astype(np.int32)

x2 = paddle.to_tensor(x)
index2 = paddle.to_tensor(index)
y2 = paddle.index_select(x2, index2, axis=-1)


x = torch.Tensor(x)
index = torch.Tensor(index).to(torch.int64)
y = torch.index_select(x, dim=-1, index=index)


print()



