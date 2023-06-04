

import numpy as np
import torch
import paddle
import torch.nn.functional as F
import paddle.nn.functional as F2

'''
代码片段来自 mmdet/models/transformers/utils.py
get_contrastive_denoising_training_group()

当第二个参数是list时，paddle填每组的size, torch也是填每组的size。
当第二个参数是int时，paddle填分组数目, torch填每组的size。

'''

num_gts = [3, 3]
max_gt_num = max(num_gts)
num_denoising = 100
num_group = num_denoising // max_gt_num



positive_gt_mask = np.zeros([2, 3]).astype(np.float32)
positive_gt_mask[0, 0] = 1.
positive_gt_mask[1, 1] = 1.
positive_gt_mask[1, 2] = 1.


positive_gt_mask2 = paddle.to_tensor(positive_gt_mask)
positive_gt_mask2 = positive_gt_mask2.tile([1, 2 * num_group])
dn_positive_idx2 = paddle.nonzero(positive_gt_mask2)
dn_positive_idx2 = dn_positive_idx2[:, 1]
aaaaaaaa = [n * num_group for n in num_gts]
dn_positive_idx2 = paddle.split(dn_positive_idx2, aaaaaaaa)

k=0

positive_gt_mask = torch.Tensor(positive_gt_mask)
positive_gt_mask = positive_gt_mask.tile([1, 2 * num_group])
dn_positive_idx = torch.nonzero(positive_gt_mask)
dn_positive_idx = dn_positive_idx[:, 1]
dn_positive_idx = torch.split(dn_positive_idx, aaaaaaaa)


print()



