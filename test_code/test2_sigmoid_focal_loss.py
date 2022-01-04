import paddle
import numpy as np
import paddle.nn.functional as F



num_positive_fp32 = 1.0
cls_logits_flatten = np.array([[-5.6, 7.3, 2.6], [13.7, -9.1, -2.3]]).astype(np.float32)
tag_labels_flatten_bin = np.array([[0, 0, 1], [0, 1, 0]]).astype(np.float32)

cls_logits_flatten222 = paddle.to_tensor(cls_logits_flatten)
tag_labels_flatten_bin222 = paddle.to_tensor(tag_labels_flatten_bin)
cls_loss222 = F.sigmoid_focal_loss(cls_logits_flatten222, tag_labels_flatten_bin222) / num_positive_fp32


import torch
cls_logits_flatten = torch.Tensor(cls_logits_flatten).cuda()
tag_labels_flatten_bin = torch.Tensor(tag_labels_flatten_bin).cuda()

def sigmoid_focal_loss(logits, labels, alpha=0.25, gamma=2.0, eps=1e-9):
    p = torch.sigmoid(logits)
    pos_loss = labels * (0 - torch.log(p + eps)) * torch.pow(1 - p, gamma) * alpha
    neg_loss = (1.0 - labels) * (0 - torch.log(1 - p + eps)) * torch.pow(p, gamma) * (1 - alpha)
    focal_loss = pos_loss + neg_loss
    return focal_loss.sum()


cls_loss22222 = sigmoid_focal_loss(cls_logits_flatten, tag_labels_flatten_bin) / num_positive_fp32

print()









