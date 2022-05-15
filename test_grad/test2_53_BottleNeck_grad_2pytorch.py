import pickle
import six
import torch
from mmdet.models import BottleNeck


ch_in = 64
ch_out = 64
stride = 1
shortcut = False
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 256
ch_out = 64
stride = 1
shortcut = True
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 256
ch_out = 64
stride = 1
shortcut = True
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 256
ch_out = 128
stride = 2
shortcut = False
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 512
ch_out = 128
stride = 1
shortcut = True
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 512
ch_out = 128
stride = 1
shortcut = True
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 512
ch_out = 128
stride = 1
shortcut = True
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 512
ch_out = 256
stride = 2
shortcut = False
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 1024
ch_out = 256
stride = 1
shortcut = True
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 1024
ch_out = 256
stride = 1
shortcut = True
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 1024
ch_out = 256
stride = 1
shortcut = True
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 1024
ch_out = 256
stride = 1
shortcut = True
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 1024
ch_out = 256
stride = 1
shortcut = True
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 1024
ch_out = 512
stride = 2
shortcut = False
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 2048
ch_out = 512
stride = 1
shortcut = True
variant = 'd'
groups = 1
base_width = 64
lr = 1.0
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False

ch_in = 2048
ch_out = 512
stride = 1
shortcut = True
variant = 'd'
groups = 1
base_width = 64
lr = 0.5
norm_type = 'bn'
norm_decay = 0.0
freeze_norm = False
dcn_v2 = False
std_senet = False


model = BottleNeck(ch_in=ch_in,
                                ch_out=ch_out,
                                stride=stride,
                                shortcut=shortcut,
                                variant=variant,
                                groups=groups,
                                base_width=base_width,
                                lr=lr,
                                norm_type=norm_type,
                                norm_decay=norm_decay,
                                freeze_norm=freeze_norm,
                                dcn_v2=dcn_v2,
                                std_senet=std_senet,
                                )
model.train()
model_std = model.state_dict()

def copy(name, w, std):
    if isinstance(w, dict):
        print()
    value2 = torch.Tensor(w)
    value = std[name]
    value.copy_(value2)
    std[name] = value


ckpt_file = '53_00.pdparams'
save_name = '53_00.pth'

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



ckpt_file = '53_08.pdparams'
save_name = '53_08_paddle.pth'

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




