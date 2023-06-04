

import numpy as np
import torch
import paddle
import torch.nn.functional as F
import paddle.nn.functional as F2

'''
代码片段来自 mmdet/models/transformers/utils.py
get_contrastive_denoising_training_group()


'''



def get_contrastive_denoising_training_group2(targets,
                                             num_classes,
                                             num_queries,
                                             class_embed,
                                             num_denoising=100,
                                             label_noise_ratio=0.5,
                                             box_noise_scale=1.0):
    if num_denoising <= 0:
        return None, None, None, None
    num_gts = [len(t) for t in targets["gt_class"]]
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None

    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group

    bs = len(targets["gt_class"])
    input_query_class = paddle.full([bs, max_gt_num], num_classes, dtype='int32')
    input_query_bbox = paddle.zeros([bs, max_gt_num, 4])
    pad_gt_mask = paddle.zeros([bs, max_gt_num])
    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets["gt_class"][i].squeeze(-1)
            input_query_bbox[i, :num_gt] = targets["gt_bbox"][i]
            pad_gt_mask[i, :num_gt] = 1


    # each group has positive and negative queries.
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    # positive and negative mask
    negative_gt_mask = paddle.zeros([bs, max_gt_num * 2, 1])
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    # nonzero() 返回[?, d], d是positive_gt_mask的维数，这里是2。 nonzero() 返回 positive_gt_mask 里非0值的坐标
    dn_positive_idx = paddle.nonzero(positive_gt_mask)
    dn_positive_idx = dn_positive_idx[:, 1]   # 只取维度1的坐标
    dn_positive_idx = paddle.split(dn_positive_idx, [n * num_group for n in num_gts])
    return dn_positive_idx



def get_contrastive_denoising_training_group(targets,
                                             num_classes,
                                             num_queries,
                                             class_embed,
                                             num_denoising=100,
                                             label_noise_ratio=0.5,
                                             box_noise_scale=1.0):
    if num_denoising <= 0:
        return None, None, None, None
    num_gts = targets["pad_gt_mask"].sum([1, 2]).to(torch.int32).cpu().detach().numpy().tolist()
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None

    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group
    # 原版对每张图片的gt pad到 max_gt_num, 但是我已经在数据预处理阶段 pad了，所以不需要做
    # bs = len(targets["gt_class"])
    # input_query_class = paddle.full([bs, max_gt_num], num_classes, dtype='int32')
    # input_query_bbox = paddle.zeros([bs, max_gt_num, 4])
    # pad_gt_mask = paddle.zeros([bs, max_gt_num])
    # for i in range(bs):
    #     num_gt = num_gts[i]
    #     if num_gt > 0:
    #         input_query_class[i, :num_gt] = targets["gt_class"][i].squeeze(-1)
    #         input_query_bbox[i, :num_gt] = targets["gt_bbox"][i]
    #         pad_gt_mask[i, :num_gt] = 1
    input_query_class = targets["gt_class"].squeeze(-1)
    input_query_bbox = targets["gt_bbox"]
    pad_gt_mask = targets["pad_gt_mask"].squeeze(-1)
    bs = input_query_bbox.shape[0]
    device = input_query_bbox.device


    # each group has positive and negative queries.
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    # positive and negative mask
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    # nonzero() 返回[?, d], d是positive_gt_mask的维数，这里是2。 nonzero() 返回 positive_gt_mask 里非0值的坐标
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]   # 只取维度1的坐标
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
    return dn_positive_idx





num_queries = 300
num_classes = 80
class_embed = None

gt_class0 = np.zeros([1, 1]).astype(np.int32)
gt_class1 = np.zeros([3, 1]).astype(np.int32)
gt_bbox0 = np.zeros([1, 4]).astype(np.float32)
gt_bbox1 = np.zeros([3, 4]).astype(np.float32)



targets ={}
targets['gt_class'] = [paddle.to_tensor(gt_class0), paddle.to_tensor(gt_class1)]
targets['gt_bbox'] = [paddle.to_tensor(gt_bbox0), paddle.to_tensor(gt_bbox1)]



aaaaaaaa1 = get_contrastive_denoising_training_group2(targets,
                                             num_classes,
                                             num_queries,
                                             class_embed)



gt_bbox = np.zeros([2, 3, 4]).astype(np.float32)
gt_class = np.zeros([2, 3, 1]).astype(np.float32)
pad_gt_mask = np.zeros([2, 3, 1]).astype(np.float32)

gt_bbox = torch.Tensor(gt_bbox)
gt_class = torch.Tensor(gt_class)
pad_gt_mask = torch.Tensor(pad_gt_mask)
pad_gt_mask[0, 0, :] = 1.
pad_gt_mask[1, :, :] = 1.



targets ={}
targets['gt_class'] = gt_class
targets['gt_bbox'] = gt_bbox
targets['pad_gt_mask'] = pad_gt_mask



aaaaaaaa2 = get_contrastive_denoising_training_group(targets,
                                             num_classes,
                                             num_queries,
                                             class_embed)





print()



