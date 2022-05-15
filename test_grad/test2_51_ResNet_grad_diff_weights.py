
import torch
import numpy as np

ckpt_file1 = '51_00.pth'
ckpt_file1 = '51_08_paddle.pth'
ckpt_file2 = '51_08.pth'


state_dict1_pytorch = torch.load(ckpt_file1, map_location=torch.device('cpu'))
state_dict2_pytorch = torch.load(ckpt_file2, map_location=torch.device('cpu'))


d_value = 0.000005
print('======================== diff(weights) > d_value=%.6f ========================' % d_value)
for key, value1 in state_dict1_pytorch.items():
    if 'num_batches_tracked' in key:
        continue
    if 'augment_pipe.' in key:
        continue
    v1 = value1.cpu().detach().numpy()
    value2 = state_dict2_pytorch[key]
    v2 = value2.cpu().detach().numpy()
    ddd = np.sum((v1 - v2) ** 2)
    if ddd > d_value:
        print('diff=%.6f (%s)' % (ddd, key))

print()
print()
print('======================== diff(weights) <= d_value=%.6f ========================' % d_value)
for key, value1 in state_dict1_pytorch.items():
    if 'num_batches_tracked' in key:
        continue
    if 'augment_pipe.' in key:
        continue
    v1 = value1.cpu().detach().numpy()
    value2 = state_dict2_pytorch[key]
    v2 = value2.cpu().detach().numpy()
    ddd = np.sum((v1 - v2) ** 2)
    if ddd <= d_value:
        print('diff=%.6f (%s)' % (ddd, key))