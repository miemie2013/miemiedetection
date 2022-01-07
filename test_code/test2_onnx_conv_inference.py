import torch
import numpy as np
import torch.nn.functional as F


'''
测试conv导出为ONNX
'''




dic2 = np.load('onnx_conv.npz')

img = dic2['x']
w = dic2['w']
b = dic2['b']
out2 = dic2['out']

model = 'conv.onnx'
import onnx
import onnxruntime

onnx_model = onnx.load(model)
graph = onnx_model.graph
node = graph.node

print('============== ONNX Model ==============')
for i in range(len(node)):
    print(node[i])
    print('------------------------------------------------')

session = onnxruntime.InferenceSession(model)
# aaaaaaaaa = session.get_inputs()

ort_inputs = {session.get_inputs()[0].name: img}
# ort_inputs = {session.get_inputs()[0].name: img, session.get_inputs()[1].name: im_size}
outputs = session.run(None, ort_inputs)
output = outputs[0]

ddd = np.sum((output - out2)**2)
print('ddd=%.6f' % ddd)
print()









