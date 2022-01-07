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



import onnx
from onnx import helper
from onnx import TensorProto
import numpy as np

X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 5, 5])
vid = 2

W_id = '%d'%vid
W = helper.make_tensor(W_id, TensorProto.FLOAT, [2, 3, 3, 3], np.reshape(w, (-1, )))
vid += 1


B_id = '%d'%vid
B = helper.make_tensor(B_id, TensorProto.FLOAT, [2], b)
vid += 1


Y_id = '%d'%vid
Y = helper.make_tensor_value_info(Y_id, TensorProto.FLOAT, [1, 2, 5, 5])
vid += 1



node_def = helper.make_node(
    'Conv',  # node name
    ['X', W_id, B_id],
    [Y_id],  # outputs
    # attributes
    strides=[1, 1],
    pads=[1, 1, 1, 1],
    kernel_shape=[3, 3],
    dilations=[1, 1],
)

graph_def = helper.make_graph(
    [node_def],
    'test_conv_mode',
    [X],  # graph inputs
    [Y],  # graph outputs
    initializer=[W, B],
)

mode_def = helper.make_model(graph_def, producer_name='onnx-example')
onnx.checker.check_model(mode_def)
onnx.save(mode_def, "./Conv.onnx")


print()









