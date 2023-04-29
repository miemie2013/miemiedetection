

import numpy as np
import torch
from mmdet.models.ops import gather_1d

'''
PositionAssigner 中使用
        假设batch_size=2, 有num_max_boxes=2个gt，A=3个预测框。 
'''

batch_size = 2
batch_size = 1
num_max_boxes = 2
num_anchors = 3
bg_index = 80
num_classes = 80


gt_labels = np.array([[[2], [3]], [[4], [5]]]).astype(np.int64)
gt_labels = torch.Tensor(gt_labels).to(torch.int64)

gt_bboxes = np.array([[[2, 3, 4, 5], [6, 7, 8, 9]], [[12, 13, 14, 15], [16, 17, 18, 19]]]).astype(np.float32)
gt_bboxes = torch.Tensor(gt_bboxes).to(torch.float32)


ious = np.array([[[0.93, 0, 0], [0.1, 0.98, 0]]]).astype(np.float32)
ious = torch.Tensor(ious).to(torch.float32)
ious = ious.repeat([batch_size, 1, 1])

cost = np.array([[[0.07, 1, 123456.], [0.9, 0.02, 123451.]]]).astype(np.float32)
cost = torch.Tensor(cost).to(torch.float32)
cost = cost.repeat([batch_size, 1, 1])  # [N, A, 4],  cost
matched_gt_cost, matched_gt_index = cost.min(dim=1)
neg_flag = matched_gt_cost > 5000.
neg_mask = neg_flag.float().unsqueeze(-1)
pos_mask = (1. - neg_mask)

batch_ind = torch.arange(end=batch_size, dtype=gt_labels.dtype).unsqueeze(-1)

matched_gt_index = matched_gt_index + batch_ind * num_max_boxes

matched_gt_index_ = matched_gt_index.flatten().to(torch.int64)
assigned_labels = gather_1d(gt_labels.flatten(), index=matched_gt_index_)
assigned_labels = assigned_labels.reshape([batch_size, num_anchors])  # assigned_labels.shape = [N, A]
assigned_labels[neg_flag] = bg_index

assigned_bboxes = gather_1d(gt_bboxes.reshape([-1, 4]), index=matched_gt_index_)
assigned_bboxes = assigned_bboxes.reshape([batch_size, num_anchors, 4])  # assigned_bboxes.shape = [N, A, 4]
assigned_bboxes = 0. * neg_mask + (1. - neg_mask) * assigned_bboxes

assigned_scores = gather_1d(gt_bboxes.reshape([-1, 4]), index=matched_gt_index_)


aaaaaaa = gt_labels[matched_gt_index]

print()



