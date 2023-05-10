
import numpy as np
import torch
import torch.nn.functional as F
import time


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

    # 对每个gt，取cost最小的k个候选正样本去学习。
    max_k = dynamic_ks.max()
    masks = torch.ones((max_k, max_k), dtype=torch.uint8, device=cost.device).tril(diagonal=0)   # [max_k, max_k]
    fill_value = masks[(dynamic_ks - 1).long(), :]   # [num_gt, max_k]   每个gt要填入 matching_matrix[num_gt, M]  的值
    _, pos_idx = torch.topk(cost, k=max_k, largest=False)   # [num_gt, max_k]   每个gt前max_k个cost最小的下标
    M = cost.shape[1]
    offset = torch.arange(start=0, end=M*num_gt, step=M, dtype=torch.int64, device=cost.device).unsqueeze(-1)  # [num_gt, 1]
    pos_idx_1d = (pos_idx + offset).flatten()   # [num_gt*max_k, ]
    matching_matrix = matching_matrix.flatten()
    matching_matrix[pos_idx_1d] = fill_value.flatten()
    matching_matrix = matching_matrix.reshape(cost.shape)
    del topk_ious, dynamic_ks, max_k, masks, fill_value, pos_idx, offset, pos_idx_1d

    # [M, ]  M个候选正样本匹配的gt数
    anchor_matching_gt = matching_matrix.sum(0)
    # deal with the case that one anchor matches multiple ground-truths
    if anchor_matching_gt.max() > 1:
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

num_gt = 2
M = 12
# num_gt = 27
# M = 283

costs1 = []
costs2 = []
for i in range(1000):
    # print('==================== test %d =====================' % (i+1,))
    cost = np.random.random((num_gt, M)).astype(np.float32)
    pair_wise_ious = np.random.random((num_gt, M)).astype(np.float32)
    # 保留2位小数
    # cost = np.around(cost, decimals=2)
    # pair_wise_ious = np.around(pair_wise_ious, decimals=2)
    # print(cost)
    # print(pair_wise_ious)
    cost = torch.Tensor(cost).to(torch.float32)
    pair_wise_ious = torch.Tensor(pair_wise_ious).to(torch.float32)
    cost = cost.cuda()
    pair_wise_ious = pair_wise_ious.cuda()


    train_start1 = time.time()
    matching_matrix1 = simota_matching(cost, pair_wise_ious, num_gt)
    cost1 = time.time() - train_start1

    train_start2 = time.time()
    matching_matrix2 = simota_matching2(cost, pair_wise_ious, num_gt)
    cost2 = time.time() - train_start2


    matching_matrix1 = matching_matrix1.cpu().detach().numpy().astype(np.float32)
    matching_matrix2 = matching_matrix2.cpu().detach().numpy().astype(np.float32)
    ddd = np.sum((matching_matrix1 - matching_matrix2) ** 2)
    assert ddd < 0.0001
    costs1.append(cost1)
    costs2.append(cost2)
    # print('ddd=%.6f' % ddd)
    # print('cost1=%.6f s, cost2=%.6f s' % (cost1, cost2))

costs1 = np.array(costs1)
costs2 = np.array(costs2)
costs1 = np.mean(costs1)
costs2 = np.mean(costs2)
print('costs1=%.6f s, costs2=%.6f s' % (costs1, costs2))

print()



