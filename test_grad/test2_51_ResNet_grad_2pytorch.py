import pickle
import six
import torch
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


model = ResNet(depth=depth, variant=variant, return_idx=return_idx, dcn_v2_stages=dcn_v2_stages,
               freeze_at=freeze_at, freeze_norm=freeze_norm, norm_decay=norm_decay)
model.train()
model_std = model.state_dict()

def copy(name, w, std):
    if isinstance(w, dict):
        print()
    value2 = torch.Tensor(w)
    value = std[name]
    value.copy_(value2)
    std[name] = value


ckpt_file = '51_00.pdparams'
save_name = '51_00.pth'

with open(ckpt_file, 'rb') as f:
    model_dic = pickle.load(f) if six.PY2 else pickle.load(f, encoding='latin1')

for key in model_dic.keys():
    name2 = key
    w = model_dic[key]
    if 'StructuredToParameterName@@' in key:
        continue
    else:
        if '._mean' in key:
            name2 = name2.replace('._mean', '.running_mean')
        if '._variance' in key:
            name2 = name2.replace('._variance', '.running_var')
        copy(name2, w, model_std)

model.load_state_dict(model_std)
torch.save(model_std, save_name)
print(torch.__version__)



ckpt_file = '51_08.pdparams'
save_name = '51_08_paddle.pth'

with open(ckpt_file, 'rb') as f:
    model_dic = pickle.load(f) if six.PY2 else pickle.load(f, encoding='latin1')

for key in model_dic.keys():
    name2 = key
    w = model_dic[key]
    if 'StructuredToParameterName@@' in key:
        continue
    else:
        if '._mean' in key:
            name2 = name2.replace('._mean', '.running_mean')
        if '._variance' in key:
            name2 = name2.replace('._variance', '.running_var')
        copy(name2, w, model_std)

model.load_state_dict(model_std)
torch.save(model_std, save_name)
print(torch.__version__)




