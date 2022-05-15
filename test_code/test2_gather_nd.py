

import numpy as np
import torch
import paddle
from mmdet.models.ops import gather_nd

'''

'''

dic = np.load('aa22.npz')
pred_scores = torch.Tensor(dic['pred_scores'])
gt_labels_ind = torch.Tensor(dic['gt_labels_ind']).to(torch.int64)
bbox_cls_scores = torch.Tensor(dic['bbox_cls_scores'])
bbox_cls_scores222 = gather_nd(pred_scores, gt_labels_ind)

ddd = np.sum((bbox_cls_scores222.cpu().detach().numpy() - bbox_cls_scores.cpu().detach().numpy())**2)
print('ddd=%.6f' % ddd)

print()



