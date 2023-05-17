
import torch
import numpy as np

if __name__ == "__main__":
    ckpt_file1 = 'PPYOLOEPlus_outputs/ppyoloe_plus_crn_l_voc2012/16.pth'
    aaa = torch.load(ckpt_file1, map_location=torch.device('cpu'))
    state_dict1_pytorch = dict()
    for key in ['model']:
        aa = aaa[key]
        for key2, value1 in aa.items():
            state_dict1_pytorch[key2] = value1.cpu().detach().numpy()

    ckpt_file2 = 'PPYOLOEPlus_outputs/ppyoloe_plus_crn_s_voc2012/16.pth'
    aaa = torch.load(ckpt_file2, map_location=torch.device('cpu'))
    state_dict2_pytorch = dict()
    for key in ['model']:
        aa = aaa[key]
        for key2, value1 in aa.items():
            if key2.startswith('teacher_model.'):
                state_dict2_pytorch[key2] = value1.cpu().detach().numpy()

    d_value = 0.0000001
    print('======================== diff(weights) > d_value=%.6f ========================' % d_value)
    for key, value1 in state_dict1_pytorch.items():
        if 'num_batches_tracked' in key:
            continue
        v1 = value1
        value2 = state_dict2_pytorch['teacher_model.' + key]
        v2 = value2
        ddd = np.sum((v1 - v2) ** 2)
        if ddd > d_value:
            print('diff=%.6f (%s)' % (ddd, key))

    print()
    print()





