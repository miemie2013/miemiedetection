

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

pred_corners = np.random.normal(size=[2, 2, 2, 2])

print(pred_corners)

dropout = 0.9
training = True

weights2 = paddle.to_tensor(pred_corners)

aaaaaaa1 = F2.dropout(
    weights2,
    dropout,
    training=training,
    mode="upscale_in_train")

weights = torch.Tensor(pred_corners)

aaaaaaa2 = F.dropout(
    weights,
    dropout,
    training=training)


aaaaaaa1 = aaaaaaa1.numpy()
aaaaaaa2 = aaaaaaa2.cpu().detach().numpy()


print(aaaaaaa1)
print(aaaaaaa2)
print()



