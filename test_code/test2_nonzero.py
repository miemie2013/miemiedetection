

import numpy as np
import torch
import paddle
import torch.nn.functional as F
import paddle.nn.functional as F2

'''
代码片段来自 mmdet/models/transformers/utils.py
get_contrastive_denoising_training_group()

nonzero() 返回[?, d], d是positive_gt_mask的维数，这里是2。 nonzero() 返回 positive_gt_mask 里非0值的坐标
paddle.nonzero() 和 torch.nonzero()  同义。
'''

positive_gt_mask = np.zeros([2, 3]).astype(np.float32)
positive_gt_mask[0, 0] = 1.
positive_gt_mask[1, :] = 1.


positive_gt_mask2 = paddle.to_tensor(positive_gt_mask)

dn_positive_idx2 = paddle.nonzero(positive_gt_mask2)
dn_positive_idx2 = dn_positive_idx2[:, 1]


k=0

positive_gt_mask = torch.Tensor(positive_gt_mask)

dn_positive_idx = torch.nonzero(positive_gt_mask)


print()



