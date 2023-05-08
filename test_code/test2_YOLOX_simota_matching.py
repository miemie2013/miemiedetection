
import numpy as np
import torch
import torch.nn.functional as F


def simota_matching(cost, pair_wise_ious, num_gt):
    '''
    为每个gt动态分配不同的正样本数。
    cost             [num_gt, M]  总的cost
    pair_wise_ious   [num_gt, M]  gt和候选正样本两两之间的iou
    gt_classes       [num_gt, ]   gt的cid
    num_gt           当前图片gt数目
    fg_mask          [A, ]     anchor至少落在1个"范围框"内时, 为True
    '''
    # [num_gt, M]  全是0，类型是uint8省内存
    matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

    n_candidate_k = min(10, pair_wise_ious.size(1))  # 选10个候选正样本，如果M < 10，选M个。下面假设M>10，n_candidate_k==10
    topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)  # [num_gt, 10]  每个gt取10个最大iou的候选正样本。
    dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)  # [num_gt, ]  iou求和作为正样本数。
    # 对每个gt，取cost最小的k个候选正样本去学习。
    for gt_idx in range(num_gt):
        _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx], largest=False)
        matching_matrix[gt_idx][pos_idx] = 1
    del topk_ious, dynamic_ks, pos_idx

    # [M, ]  M个候选正样本匹配的gt数
    anchor_matching_gt = matching_matrix.sum(0)
    # deal with the case that one anchor matches multiple ground-truths
    if anchor_matching_gt.max() > 1:
        print('anchor_matching_gt.max() > 1')
        multiple_match_mask = anchor_matching_gt > 1  # [M, ]  M个候选正样本 一对多 处为1
        matching_matrix[:, multiple_match_mask] *= 0  # 一对多的候选正样本，不匹配任何gt
        _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
        matching_matrix[cost_argmin, multiple_match_mask] = 1  # 一对多的候选正样本，匹配cost最小的gt
    fg_mask_inboxes = anchor_matching_gt > 0  # 可能有的候选正样本匹配了0个gt。将匹配多于0个gt的候选正样本作为最终正样本。
    num_fg = fg_mask_inboxes.sum().item()  # anchor前景数
    return matching_matrix


def simota_matching2(cost, pair_wise_ious, num_gt):
    '''
    为每个gt动态分配不同的正样本数。
    cost             [num_gt, M]  总的cost
    pair_wise_ious   [num_gt, M]  gt和候选正样本两两之间的iou
    gt_classes       [num_gt, ]   gt的cid
    num_gt           当前图片gt数目
    fg_mask          [A, ]     anchor至少落在1个"范围框"内时, 为True
    '''
    # [num_gt, M]  全是0，类型是uint8省内存
    matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

    n_candidate_k = min(10, pair_wise_ious.size(1))  # 选10个候选正样本，如果M < 10，选M个。下面假设M>10，n_candidate_k==10
    topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)  # [num_gt, 10]  每个gt取10个最大iou的候选正样本。
    dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)  # [num_gt, ]  iou求和作为正样本数。







    max_k = dynamic_ks.max()
    aaaaaaaaaa = torch.ones((max_k, max_k))
    aaaaaaa = aaaaaaaaaa.tril(diagonal=0)
    ii = dynamic_ks - 1
    aa = aaaaaaa[ii]

    _, pppppp = torch.topk(cost, k=max_k, largest=False)
    batch_ind = torch.arange(end=max_k, dtype=max_k.dtype).unsqueeze(0).repeat([num_gt, 1])
    dynamic_ks___ = dynamic_ks.unsqueeze(1).repeat([1, max_k])
    mmm = (batch_ind < dynamic_ks___).to(torch.uint8)



    matching_matrix[pppppp] = mmm
    print()
    # 对每个gt，取cost最小的k个候选正样本去学习。
    for gt_idx in range(num_gt):
        _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx], largest=False)
        matching_matrix[gt_idx][pos_idx] = 1
    del topk_ious, dynamic_ks, pos_idx

    # [M, ]  M个候选正样本匹配的gt数
    anchor_matching_gt = matching_matrix.sum(0)
    # deal with the case that one anchor matches multiple ground-truths
    if anchor_matching_gt.max() > 1:
        print('anchor_matching_gt.max() > 1')
        multiple_match_mask = anchor_matching_gt > 1  # [M, ]  M个候选正样本 一对多 处为1
        matching_matrix[:, multiple_match_mask] *= 0  # 一对多的候选正样本，不匹配任何gt
        _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
        matching_matrix[cost_argmin, multiple_match_mask] = 1  # 一对多的候选正样本，匹配cost最小的gt
    fg_mask_inboxes = anchor_matching_gt > 0  # 可能有的候选正样本匹配了0个gt。将匹配多于0个gt的候选正样本作为最终正样本。
    num_fg = fg_mask_inboxes.sum().item()  # anchor前景数
    return matching_matrix



num_gt = 2
M = 3


'''
[[0.32 0.43 0.06]
 [0.86 0.61 0.4 ]]
[[0.33 0.51 0.6 ]
 [0.18 0.91 0.05]]
'''
cost = np.array([[0.32, 0.43, 0.06], [0.86, 0.61, 0.40]]).astype(np.float32)
pair_wise_ious = np.array([[0.33, 0.51,  0.60], [0.18, 0.91, 0.05]]).astype(np.float32)

print(cost)
print(pair_wise_ious)
cost = torch.Tensor(cost).to(torch.float32)
pair_wise_ious = torch.Tensor(pair_wise_ious).to(torch.float32)
matching_matrix1 = simota_matching(cost, pair_wise_ious, num_gt)


print('\n\n\n')
print('====================================================================')

num_gt = 3
M = 12
for i in range(10):
    print('==================== test %d =====================' % (i+1,))
    cost = np.random.random((num_gt, M)).astype(np.float32)
    pair_wise_ious = np.random.random((num_gt, M)).astype(np.float32)
    # 保留2位小数
    cost = np.around(cost, decimals=2)
    pair_wise_ious = np.around(pair_wise_ious, decimals=2)
    print(cost)
    print(pair_wise_ious)
    cost = torch.Tensor(cost).to(torch.float32)
    pair_wise_ious = torch.Tensor(pair_wise_ious).to(torch.float32)
    matching_matrix1 = simota_matching(cost, pair_wise_ious, num_gt)
    matching_matrix2 = simota_matching2(cost, pair_wise_ious, num_gt)


print()



