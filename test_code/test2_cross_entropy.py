

import numpy as np
import torch
import paddle
import torch.nn.functional as F
import paddle.nn.functional as F2

'''
代码片段来自 mmdet/models/losses/detr_loss.py

'''

print(paddle.__version__)

num_classes = 80

logits = np.random.normal(size=[2, 2, num_classes])

target_label = np.zeros([2, 2]).astype(np.int64) + 80
target_label[0, 0] = 17
target_label[1, 1] = 33


logits2 = paddle.to_tensor(logits)
target_label2 = paddle.to_tensor(target_label).astype('int64')

# loss_2 = F2.cross_entropy(logits2, target_label2, weight=aaaa)
loss_2 = F2.cross_entropy(logits2, target_label2, reduction='none')
# loss_2 = loss_2.numpy()
# print(loss_2)

k=0

logits = torch.Tensor(logits)
target_label = torch.Tensor(target_label).to(torch.int64)

# [N, A, num_classes + 1]    每个anchor学习的one_hot向量
assigned_scores = F.one_hot(target_label, num_classes + 1)
assigned_scores = assigned_scores.to(torch.float32)
assigned_scores = assigned_scores[:, :, :-1]

# loss_1 = F.cross_entropy(logits, assigned_scores, reduction='none')

# loss_1 = F.cross_entropy(logits, target_label, reduction='none')


eps = 1e-9
p = F.softmax(logits, dim=-1)

pos_loss = assigned_scores * (0 - torch.log(p + eps))
pos_loss = pos_loss.sum(-1)

print()



