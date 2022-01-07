import torch
import numpy as np
import torch.nn.functional as F


'''
测试conv导出为ONNX
'''



x = torch.rand((1, 3, 5, 5))

dic = {}
dic['x'] = x.cpu().detach().numpy()

conv = torch.nn.Conv2d(
    in_channels=3,
    out_channels=2,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True)
conv.eval()
dic['w'] = conv.weight.cpu().detach().numpy()
dic['b'] = conv.bias.cpu().detach().numpy()

out = conv(x)

# 导出时不需要第二个输入im_size
dummy_input = torch.randn(1, 3, 5, 5)

torch.onnx._export(
    conv,
    dummy_input,
    'conv.onnx',
    input_names=['image0'],
    output_names=['out0'],
    dynamic_axes=None,
    opset_version=11,
)

dic['out'] = out.cpu().detach().numpy()
np.savez('onnx_conv', **dic)
print()









