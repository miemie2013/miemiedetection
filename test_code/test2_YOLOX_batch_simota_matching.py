
import numpy as np
import torch
import torch.nn.functional as F
import time


def simota_matching(cost, pair_wise_ious, pad_gt_mask):
    '''
    为每个gt动态分配不同的正样本数。
    cost             [N, G, A]  总的cost
    pair_wise_ious   [N, G, A]  gt和 所有anchor 两两之间的iou
    gt_classes       [N, G]     gt的cid
    is_in_centers    [N, G, A]  若某个格子中心点落在某个"范围框"内, 值为True
    pad_gt_mask      [N, G, 1]  是真gt还是填充的假gt, float类型
    '''
    N, G, A = cost.shape
    # [N, G, A]  全是0，类型是uint8省内存
    matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

    n_candidate_k = 3    # 选10个候选样本
    topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=2)   # [N, G, 10]  每个gt取10个最大iou的anchor。
    dynamic_ks = torch.clamp(topk_ious.sum(2).int(), min=1)           # [N, G]  iou求和作为正样本数。
    dynamic_ks *= pad_gt_mask[:, :, 0].int()    # [N, G]  假gt正样本数为0。
    # aa2 = dynamic_ks.cpu().detach().numpy()
    # 对每个gt，取cost最小的k个anchor去学习。
    for b_idx in range(N):
        for gt_idx in range(G):
            if dynamic_ks[b_idx, gt_idx] == 0:   # 假gt跳过
                continue
            _, pos_idx = torch.topk(cost[b_idx, gt_idx], k=dynamic_ks[b_idx, gt_idx], largest=False)
            matching_matrix[b_idx, gt_idx][pos_idx] = 1
    return matching_matrix


def simota_matching2(cost, pair_wise_ious, pad_gt_mask):
    '''
    为每个gt动态分配不同的正样本数。
    cost             [N, G, A]  总的cost
    pair_wise_ious   [N, G, A]  gt和 所有anchor 两两之间的iou
    gt_classes       [N, G]     gt的cid
    is_in_centers    [N, G, A]  若某个格子中心点落在某个"范围框"内, 值为True
    pad_gt_mask      [N, G, 1]  是真gt还是填充的假gt, float类型
    '''
    N, G, A = cost.shape
    # [N, G, A]  全是0，类型是uint8省内存
    matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

    n_candidate_k = 3    # 选10个候选样本
    topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=2)   # [N, G, 10]  每个gt取10个最大iou的anchor。
    dynamic_ks = torch.clamp(topk_ious.sum(2).int(), min=1)           # [N, G]  iou求和作为正样本数。
    dynamic_ks *= pad_gt_mask[:, :, 0].int()    # [N, G]  假gt正样本数为0。
    # 对每个gt，取cost最小的k个anchor去学习。
    max_k = dynamic_ks.max()
    masks = torch.ones((max_k+1, max_k+1), dtype=torch.uint8, device=cost.device).tril(diagonal=-1)
    masks = masks[:, :-1]   # [max_k+1, max_k]
    fill_value = masks[dynamic_ks.long(), :]  # [N, G, max_k]
    _, pos_idx = torch.topk(cost, k=max_k, largest=False)   # [N, G, max_k]
    offset = torch.arange(start=0, end=N * G * A, step=A, dtype=torch.int64, device=cost.device).unsqueeze(-1)  # [N*G, 1]
    pos_idx = pos_idx.reshape([N*G, -1])       # [N*G, max_k]
    pos_idx_1d = (pos_idx + offset).flatten()  # [N*G*max_k, ]
    matching_matrix = matching_matrix.flatten()
    matching_matrix[pos_idx_1d] = fill_value.flatten()
    matching_matrix = matching_matrix.reshape(cost.shape)
    return matching_matrix



N = 2
G = 2
A = 7

pad_gt_mask = np.ones((N, G, 1)).astype(np.float32)
pad_gt_mask[0][1][0] = 0.
pad_gt_mask = torch.Tensor(pad_gt_mask).to(torch.float32)
pad_gt_mask = pad_gt_mask.cuda()


costs1 = []
costs2 = []
for i in range(1000):
    print('==================== test %d =====================' % (i+1,))
    cost = np.random.random((N, G, A)).astype(np.float32)
    pair_wise_ious = np.random.random((N, G, A)).astype(np.float32)
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
    matching_matrix1 = simota_matching(cost, pair_wise_ious, pad_gt_mask)
    cost1 = time.time() - train_start1

    train_start2 = time.time()
    matching_matrix2 = simota_matching2(cost, pair_wise_ious, pad_gt_mask)
    cost2 = time.time() - train_start2


    matching_matrix1 = matching_matrix1.cpu().detach().numpy().astype(np.float32)
    matching_matrix2 = matching_matrix2.cpu().detach().numpy().astype(np.float32)
    ddd = np.sum((matching_matrix1 - matching_matrix2) ** 2)
    if ddd > 0.0001:
        cccccccc = cost.cpu().detach().numpy().astype(np.float32)
        print()
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



