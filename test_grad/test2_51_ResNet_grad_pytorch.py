
import torch
import numpy as np
from mmdet.models import ResNet


depth = 50
variant = 'd'
return_idx = [1, 2, 3]
dcn_v2_stages = [-1]
freeze_at = -1
freeze_norm = False
norm_decay = 0.

depth = 50
variant = 'd'
return_idx = [1, 2, 3]
dcn_v2_stages = [-1]
freeze_at = 2
freeze_norm = False
norm_decay = 0.





torch.backends.cudnn.benchmark = True  # Improves training speed.
torch.backends.cuda.matmul.allow_tf32 = False  # Allow PyTorch to internally use tf32 for matmul
torch.backends.cudnn.allow_tf32 = False  # Allow PyTorch to internally use tf32 for convolutions

model = ResNet(depth=depth, variant=variant, return_idx=return_idx, dcn_v2_stages=dcn_v2_stages,
               freeze_at=freeze_at, freeze_norm=freeze_norm, norm_decay=norm_decay)
model.train()
need_clip = False

base_lr = 0.00000001 * 1.0
param_groups = []
base_wd = 0.0005
# base_wd = 0.0
momentum = 0.9
# 是否进行梯度裁剪
need_clip = False
clip_norm = 1000000.0
# need_clip = True
# clip_norm = 35.0
model.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
optimizer = torch.optim.SGD(param_groups, lr=base_lr, momentum=momentum, weight_decay=base_wd)
model.load_state_dict(torch.load("51_00.pth", map_location=torch.device('cpu')))
model.fix_bn()

use_gpu = True
if use_gpu:
    model = model.cuda()

dic2 = np.load('51.npz')
print(torch.__version__)
for batch_idx in range(8):
    print('======================== batch_%.3d ========================'%batch_idx)
    optimizer.zero_grad(set_to_none=True)
    x = dic2['batch_%.3d.x'%batch_idx]
    y_52x52_paddle = dic2['batch_%.3d.y_52x52'%batch_idx]
    y_26x26_paddle = dic2['batch_%.3d.y_26x26'%batch_idx]
    y_13x13_paddle = dic2['batch_%.3d.y_13x13'%batch_idx]
    # w_grad_paddle = dic2['batch_%.3d.w_grad'%batch_idx]
    # b_grad_paddle = dic2['batch_%.3d.b_grad'%batch_idx]

    x = torch.Tensor(x)
    if use_gpu:
        x = x.cuda()
    x.requires_grad_(True)

    y = model(x)
    y_52x52 = y[0]
    y_26x26 = y[1]
    y_13x13 = y[2]

    y_52x52_pytorch = y_52x52.cpu().detach().numpy()
    ddd = np.sum((y_52x52_pytorch - y_52x52_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    y_26x26_pytorch = y_26x26.cpu().detach().numpy()
    ddd = np.sum((y_26x26_pytorch - y_26x26_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    y_13x13_pytorch = y_13x13.cpu().detach().numpy()
    ddd = np.sum((y_13x13_pytorch - y_13x13_paddle) ** 2)
    print('ddd=%.6f' % ddd)

    loss = y_13x13.sum()
    loss.backward()

    # 注意，这是裁剪之前的梯度，Paddle无法获得裁剪后的梯度。
    # w_grad_pytorch = model.stage5_2.conv3.conv.weight.cpu().detach().numpy()
    # ddd = np.sum((w_grad_pytorch - w_grad_paddle) ** 2)
    # print('ddd=%.6f' % ddd)
    #
    # b_grad_pytorch = model.stage5_2.conv3.bn.bias.cpu().detach().numpy()
    # ddd = np.sum((b_grad_pytorch - b_grad_paddle) ** 2)
    # print('ddd=%.6f' % ddd)

    # 梯度裁剪
    if need_clip:
        for param_group in optimizer.param_groups:
            if param_group['need_clip']:
                torch.nn.utils.clip_grad_norm_(param_group['params'], max_norm=param_group['clip_norm'], norm_type=2)
    optimizer.step()
torch.save(model.state_dict(), "51_08.pth")
print(torch.__version__)
print()
