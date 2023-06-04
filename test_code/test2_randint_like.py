

import numpy as np
import torch
import paddle
import torch.nn.functional as F
import paddle.nn.functional as F2

'''
代码片段来自 mmdet/models/transformers/utils.py
get_contrastive_denoising_training_group()

nonzero() 返回[?, d], d是positive_gt_mask的维数，这里是2。 nonzero() 返回 positive_gt_mask 里非0值的坐标
paddle.randint_like() 和 torch.randint_like()  同义。
'''
num_classes = 80


chosen_idx = np.zeros([6, ]).astype(np.int32)
chosen_idx[0] = 37
chosen_idx[1] = 76


chosen_idx2 = paddle.to_tensor(chosen_idx)
chosen_idx2 = chosen_idx2.astype('int64')

# new_label = paddle.randint_like(chosen_idx2, 0, num_classes, dtype=paddle.int64)
k=0

chosen_idx = torch.Tensor(chosen_idx)


new_label = torch.randint_like(chosen_idx, 0, num_classes, dtype=torch.int64)


print()



