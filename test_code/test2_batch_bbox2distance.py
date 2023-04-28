

import numpy as np
import torch
from mmdet.models.ops import gather_1d

'''
PositionAssigner 中使用
        centers              [A, 2]  先验框中心点坐标（单位是像素）
        pred_bboxes          [N, A, 4],  预测框左上角坐标、右下角坐标；单位是像素
'''

centers = np.array([[0.5, 0.5], [1.5, 0.5]]).astype(np.float32)
centers = torch.Tensor(centers).to(torch.float32)

pred_bboxes = np.array([[0.1, 0.2, 0.7, 0.9], [1.1, 0.3, 1.9, 0.8]]).astype(np.float32)
pred_bboxes = torch.Tensor(pred_bboxes).to(torch.float32)
N = 1
pred_bboxes = pred_bboxes.unsqueeze(0).repeat([N, 1, 1])  # [N, A, 4],  预测框左上角坐标、右下角坐标；单位是像素


x1y1, x2y2 = torch.split(pred_bboxes, 2, -1)  # x1y1.shape == [N, A, 2],  x2y2.shape == [N, A, 2]


lt = centers - x1y1  # lt.shape == [N, A, 2],  预测的lt
rb = x2y2 - centers  # rb.shape == [N, A, 2],  预测的rb

l = lt[:, :, 0]
t = lt[:, :, 1]
r = rb[:, :, 0]
b = rb[:, :, 1]

centerness = torch.min(l, r) * torch.min(t, b) / (torch.max(l, r) * torch.max(t, b))
centerness = centerness.sqrt()


print()



