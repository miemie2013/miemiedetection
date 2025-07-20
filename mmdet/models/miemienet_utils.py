import struct
import numpy as np
import math
import torch
import torch.nn as nn

convert_to_fp16 = False

def set_convert_to_fp16(_convert_to_fp16=False):
    global convert_to_fp16
    convert_to_fp16 = _convert_to_fp16



class MMTensor:
    def __init__(self, tensor_id, tensor_type='t'):
        super().__init__()
        self.tensor_id = tensor_id
        assert tensor_type in ['i', 't', 'o']
        self.tensor_type = tensor_type  # i表示是网络输入张量，t表示是网络中间张量，o表示是网络输出张量

class MMLayer:
    def __init__(self, layer_id, layer_type='', layer_arg=''):
        super().__init__()
        # {层id}:{层类型}:{层参数，如果有多个就用,分割}:{层输入tensor_id，带=i表示是网络输入张量，带=t表示是网络中间张量，如果有多个就用,分割}:{层输出tensor_id，带=o表示是网络输出张量，带=t表示是网络中间张量，如果有多个就用,分割}
        self.layer_id = layer_id
        self.layer_type = layer_type
        self.layer_arg = layer_arg
        self.input_tensors = []
        self.output_tensors = []

    def add_input_tensor(self, x):
        self.input_tensors.append(x)

    def add_output_tensor(self, x):
        self.output_tensors.append(x)


def bp_write_value(bp, value, force_fp32=True):
    if force_fp32:
        s = struct.pack('f', value)
    else:
        raise NotImplementedError("fp16 is not implemented.")
    bp.write(s)
    return bp

def bp_write_ndarray(bp, ndarray):
    assert isinstance(ndarray, np.ndarray)
    ndarray = np.reshape(ndarray, (-1,))
    for i1 in range(ndarray.shape[0]):
        bp = bp_write_value(bp, ndarray[i1], force_fp32=True)
    return bp


# 创建张量
def create_tensor(mm_data, tensor_type='t'):
    # i表示是网络输入张量，t表示是网络中间张量，o表示是网络输出张量
    assert tensor_type in ['i', 't', 'o']
    tensor_id = mm_data['tensor_id']
    mm_data['tensor_id'] = tensor_id + 1
    tensor = MMTensor(tensor_id, tensor_type=tensor_type)
    return tensor

# 标记张量为输出张量
def mark_tensor_as_output(mm_data, tensor):
    if isinstance(tensor, MMTensor):
        tensor.tensor_type = 'o'
    elif isinstance(tensor, list):
        for te in tensor:
            assert isinstance(te, MMTensor)
            te.tensor_type = 'o'

# 导出网络结构文件
def export_net(mm_data):
    # mm_data里的tensor也跟着会修改 tensor_type = 'o'
    save_name = mm_data['save_name']
    layers = mm_data['layers']
    content = '%d\n' % len(layers)
    for layer in layers:
        in_content = ''
        out_content = ''
        for tens in layer.input_tensors:
            in_content += '%d=%s,' % (tens.tensor_id, tens.tensor_type)
        for tens in layer.output_tensors:
            out_content += '%d=%s,' % (tens.tensor_id, tens.tensor_type)
        line = '%d:%s:%s:%s:%s\n' % (layer.layer_id, layer.layer_type, layer.layer_arg, in_content, out_content)
        content += line
    with open('%s.mie' % save_name, 'w', encoding='utf-8') as f:
        f.write(content)
        f.close()


# 创建 神经网络层 并且 forward，外部不要调用，而是应该调用 create_layer_and_forward()
def get_str(ndarray, max_num=7):
    assert isinstance(ndarray, np.ndarray)
    arr2 = np.copy(ndarray)
    arr2 = np.reshape(arr2, (-1,))
    st = ''
    for i in range(max_num):
        st += '%f,' % arr2[i]
    return st


# 创建 神经网络层 并且 forward，外部不要调用，而是应该调用 create_layer_and_forward()
def create_leakyrelu_and_forward(mm_data, input_tensors, layer_args):
    assert isinstance(input_tensors, MMTensor)
    assert isinstance(layer_args, dict)
    assert 'negative_slope' in layer_args.keys()
    negative_slope = layer_args['negative_slope']

    layer_id = mm_data['layer_id']
    mm_data['layer_id'] = layer_id + 1

    # 获取 layer 参数

    # 创建层
    layer_arg = 'negative_slope=%f,' % (negative_slope, )
    layer = MMLayer(layer_id, 'LeakyReLU', layer_arg)
    # 层添加输入张量
    layer.add_input_tensor(input_tensors)
    # 创建输出张量
    output = create_tensor(mm_data)
    # 层添加输出张量
    layer.add_output_tensor(output)
    # mm_data中添加层
    mm_data['layers'].append(layer)

    return output

# 创建 神经网络层 并且 forward，外部不要调用，而是应该调用 create_layer_and_forward()
def create_Softmax_and_forward(mm_data, input_tensors, layer_args):
    assert isinstance(input_tensors, MMTensor)
    assert isinstance(layer_args, dict)
    assert 'dim' in layer_args.keys()
    dim = layer_args['dim']

    layer_id = mm_data['layer_id']
    mm_data['layer_id'] = layer_id + 1

    # 获取 layer 参数

    # 创建层
    layer_arg = 'dim=%d,' % (dim, )
    layer = MMLayer(layer_id, 'Softmax', layer_arg)
    # 层添加输入张量
    layer.add_input_tensor(input_tensors)
    # 创建输出张量
    output = create_tensor(mm_data)
    # 层添加输出张量
    layer.add_output_tensor(output)
    # mm_data中添加层
    mm_data['layers'].append(layer)

    return output

# 创建 神经网络层 并且 forward，外部不要调用，而是应该调用 create_layer_and_forward()
def create_Transpose_and_forward(mm_data, input_tensors, layer_args):
    assert isinstance(input_tensors, MMTensor)
    assert isinstance(layer_args, dict)
    assert 'trans_type' in layer_args.keys()
    _trans_type = layer_args['trans_type']

    trans_type = -1
    if _trans_type == '0213':
        trans_type = 0
    elif _trans_type == '0312':
        trans_type = 1
    else:
        raise NotImplementedError("not implemented.")

    layer_id = mm_data['layer_id']
    mm_data['layer_id'] = layer_id + 1

    # 获取 layer 参数

    # 创建层
    layer_arg = 'trans_type=%d,' % (trans_type, )
    layer = MMLayer(layer_id, 'Transpose', layer_arg)
    # 层添加输入张量
    layer.add_input_tensor(input_tensors)
    # 创建输出张量
    output = create_tensor(mm_data)
    # 层添加输出张量
    layer.add_output_tensor(output)
    # mm_data中添加层
    mm_data['layers'].append(layer)

    return output


# 创建 神经网络层 并且 forward，外部不要调用，而是应该调用 create_layer_and_forward()
def create_Reshape_and_forward(mm_data, input_tensors, layer_args):
    assert isinstance(input_tensors, MMTensor)
    assert isinstance(layer_args, dict)
    dim0 = layer_args.get('dim0', -2)
    dim1 = layer_args.get('dim1', -2)
    dim2 = layer_args.get('dim2', -2)
    dim3 = layer_args.get('dim3', -2)
    dim4 = layer_args.get('dim4', -2)
    dim5 = layer_args.get('dim5', -2)

    layer_id = mm_data['layer_id']
    mm_data['layer_id'] = layer_id + 1

    # 获取 layer 参数

    # 创建层
    layer_arg = 'dim0=%d,dim1=%d,dim2=%d,dim3=%d,dim4=%d,dim5=%d,' % (dim0, dim1, dim2, dim3, dim4, dim5)
    layer = MMLayer(layer_id, 'Reshape', layer_arg)
    # 层添加输入张量
    layer.add_input_tensor(input_tensors)
    # 创建输出张量
    output = create_tensor(mm_data)
    # 层添加输出张量
    layer.add_output_tensor(output)
    # mm_data中添加层
    mm_data['layers'].append(layer)

    return output


# 创建 神经网络层 并且 forward，外部不要调用，而是应该调用 create_layer_and_forward()
def create_AvgPool2d_str_and_forward(mm_data, input_tensors, layer_args):
    assert isinstance(input_tensors, MMTensor)
    assert isinstance(layer_args, dict)
    assert 'globelpool' in layer_args.keys()
    globelpool = layer_args['globelpool']

    layer_id = mm_data['layer_id']
    mm_data['layer_id'] = layer_id + 1

    # 获取 layer 参数

    # 创建层
    layer_arg = 'globelpool=%d,' % (globelpool, )
    layer = MMLayer(layer_id, 'AvgPool2d', layer_arg)
    # 层添加输入张量
    layer.add_input_tensor(input_tensors)
    # 创建输出张量
    output = create_tensor(mm_data)
    # 层添加输出张量
    layer.add_output_tensor(output)
    # mm_data中添加层
    mm_data['layers'].append(layer)

    return output

# 创建 神经网络层 并且 forward，外部不要调用，而是应该调用 create_layer_and_forward()
def create_ConCat_and_forward(mm_data, input_tensors, layer_args):
    assert isinstance(input_tensors, list)
    assert isinstance(input_tensors[0], MMTensor)
    assert isinstance(input_tensors[1], MMTensor)
    assert isinstance(layer_args, dict)
    assert 'dim' in layer_args.keys()
    dim = layer_args['dim']

    layer_id = mm_data['layer_id']
    mm_data['layer_id'] = layer_id + 1

    # 获取 layer 参数

    # 创建层
    layer_arg = 'dim=%d,' % (dim, )
    layer = MMLayer(layer_id, 'ConCat', layer_arg)
    # 层添加输入张量
    for intensor in input_tensors:
        layer.add_input_tensor(intensor)
    # 创建输出张量
    output = create_tensor(mm_data)
    # 层添加输出张量
    layer.add_output_tensor(output)
    # mm_data中添加层
    mm_data['layers'].append(layer)

    return output


# 创建 神经网络层 并且 forward，外部不要调用，而是应该调用 create_layer_and_forward()
def create_Constxxx_and_forward(mm_data, input_tensors, layer_args, layer_name):
    assert isinstance(input_tensors, MMTensor)
    assert isinstance(layer_args, dict)
    assert 'value' in layer_args.keys()
    value = layer_args['value']

    layer_id = mm_data['layer_id']
    mm_data['layer_id'] = layer_id + 1

    # 获取 layer 参数

    # 创建层
    layer_arg = 'value=%f,' % (value, )
    layer = MMLayer(layer_id, 'Const%s' % layer_name, layer_arg)
    # 层添加输入张量
    layer.add_input_tensor(input_tensors)
    # 创建输出张量
    output = create_tensor(mm_data)
    # 层添加输出张量
    layer.add_output_tensor(output)
    # mm_data中添加层
    mm_data['layers'].append(layer)

    return output



# 创建 神经网络层 并且 forward，外部不要调用，而是应该调用 create_layer_and_forward()
def create_act_no_args_and_forward(mm_data, input_tensors, layer_args):
    assert isinstance(input_tensors, MMTensor)
    assert isinstance(layer_args, dict)
    assert 'act_type' in layer_args.keys()
    _act_type = layer_args['act_type']

    act_type = -1
    if _act_type == 'ReLU':
        act_type = 0
    elif _act_type == 'Sigmoid':
        act_type = 1
    elif _act_type == 'SiLU':
        act_type = 2
    elif _act_type == 'Hardsigmoid':
        act_type = 3
    else:
        raise NotImplementedError("not implemented.")

    layer_id = mm_data['layer_id']
    mm_data['layer_id'] = layer_id + 1

    # 获取 layer 参数

    # 创建层
    layer_arg = 'act_type=%d,' % (act_type, )
    layer = MMLayer(layer_id, 'ActNoArgs', layer_arg)
    # 层添加输入张量
    layer.add_input_tensor(input_tensors)
    # 创建输出张量
    output = create_tensor(mm_data)
    # 层添加输出张量
    layer.add_output_tensor(output)
    # mm_data中添加层
    mm_data['layers'].append(layer)

    return output


# 创建 神经网络层 并且 forward，外部不要调用，而是应该调用 create_layer_and_forward()
def create_Interp_and_forward(mm_data, input_tensors, layer_args):
    assert isinstance(input_tensors, MMTensor)
    assert isinstance(layer_args, dict)
    assert 'size_h' in layer_args.keys()
    _size_h = layer_args['size_h']
    assert 'size_w' in layer_args.keys()
    _size_w = layer_args['size_w']
    assert 'scale_h' in layer_args.keys()
    _scale_h = layer_args['scale_h']
    assert 'scale_w' in layer_args.keys()
    _scale_w = layer_args['scale_w']
    assert 'mode' in layer_args.keys()
    _mode = layer_args['mode']
    assert 'align_corners' in layer_args.keys()
    _align_corners = layer_args['align_corners']
    assert 'recompute_scale_factor' in layer_args.keys()
    _recompute_scale_factor = layer_args['recompute_scale_factor']

    mode = -1
    if _mode == 'nearest':
        mode = 0
    elif _mode == 'bilinear':
        mode = 1
    else:
        raise NotImplementedError("not implemented.")

    layer_id = mm_data['layer_id']
    mm_data['layer_id'] = layer_id + 1

    # 获取 layer 参数

    # 创建层
    layer_arg = 'size_h=%d,size_w=%d,scale_h=%f,scale_w=%f,mode=%d,align_corners=%d,recompute_scale_factor=%d,' % (_size_h, _size_w, _scale_h, _scale_w, mode, _align_corners, _recompute_scale_factor)
    layer = MMLayer(layer_id, 'Interp', layer_arg)
    # 层添加输入张量
    layer.add_input_tensor(input_tensors)
    # 创建输出张量
    output = create_tensor(mm_data)
    # 层添加输出张量
    layer.add_output_tensor(output)
    # mm_data中添加层
    mm_data['layers'].append(layer)

    return output


# 创建 神经网络层 并且 forward，外部不要调用，而是应该调用 create_layer_and_forward()
def create_ElementWise_and_forward(mm_data, input_tensors, layer_args):
    assert isinstance(input_tensors, list)
    assert len(input_tensors) == 2
    assert isinstance(input_tensors[0], MMTensor)
    assert isinstance(input_tensors[1], MMTensor)
    assert 'op_type' in layer_args.keys()
    _op_type = layer_args['op_type']

    op_type = -1
    if _op_type == 'add':
        op_type = 0
    elif _op_type == 'sub':
        op_type = 1
    elif _op_type == 'mul':
        op_type = 2
    elif _op_type == 'div':
        op_type = 3
    else:
        raise NotImplementedError("not implemented.")

    layer_id = mm_data['layer_id']
    mm_data['layer_id'] = layer_id + 1

    # 获取 layer 参数

    # 创建层
    layer_arg = 'op_type=%d,' % (op_type, )
    layer = MMLayer(layer_id, 'ElementWise', layer_arg)
    # 层添加输入张量
    layer.add_input_tensor(input_tensors[0])
    layer.add_input_tensor(input_tensors[1])
    # 创建输出张量
    output = create_tensor(mm_data)
    # 层添加输出张量
    layer.add_output_tensor(output)
    # mm_data中添加层
    mm_data['layers'].append(layer)

    return output


# 创建 神经网络层 并且 forward，外部不要调用，而是应该调用 create_layer_and_forward()
def create_conv2d_and_forward(mm_data, torch_layer, input_tensors):
    assert isinstance(input_tensors, MMTensor)
    layer_id = mm_data['layer_id']
    mm_data['layer_id'] = layer_id + 1

    # 获取 layer 参数
    '''
    Conv2d(int in_channels, int out_channels, int kernel_h=1, int kernel_w=1, int stride_h=1, int stride_w=1, int padding_h=0, 
    int padding_w=0, int dilation_h=1, int dilation_w=1, int groups=1, bool use_bias=true, bool create_weights=true);

    '''
    # 创建层
    bias = torch_layer.bias
    use_bias = 0
    if bias is not None:
        use_bias = 1
    layer_arg = 'in_channels=%d,out_channels=%d,kernel_h=%d,kernel_w=%d,stride_h=%d,stride_w=%d,padding_h=%d,padding_w=%d,dilation_h=%d,dilation_w=%d,groups=%d,use_bias=%d,' % (
        torch_layer.in_channels, torch_layer.out_channels, torch_layer.kernel_size[0], torch_layer.kernel_size[1],
        torch_layer.stride[0], torch_layer.stride[1], torch_layer.padding[0], torch_layer.padding[1],
        torch_layer.dilation[0], torch_layer.dilation[1], torch_layer.groups, use_bias)
    layer = MMLayer(layer_id, 'Conv2d', layer_arg)
    # 层添加输入张量
    layer.add_input_tensor(input_tensors)
    # 创建输出张量
    output = create_tensor(mm_data)
    # 层添加输出张量
    layer.add_output_tensor(output)
    # mm_data中添加层
    mm_data['layers'].append(layer)

    # 权重写入权重文件
    net_weight_file = mm_data['net_weight_file']
    ndarray = torch_layer.weight.cpu().detach().numpy()

    # [out_C, in_C, kH, kW] -> [kH, kW, in_C, out_C]
    ndarray = ndarray.transpose((2, 3, 1, 0))
    # print('conv2d.weight = %s' % get_str(ndarray))

    net_weight_file = bp_write_ndarray(net_weight_file, ndarray)
    if bias is not None:
        ndarray = bias.cpu().detach().numpy()
        # print('conv2d.bias = %s' % get_str(ndarray))
        net_weight_file = bp_write_ndarray(net_weight_file, ndarray)
    return output


# 创建 神经网络层 并且 forward，外部不要调用，而是应该调用 create_layer_and_forward()
def create_MaxPool2d_and_forward(mm_data, torch_layer, input_tensors):
    assert isinstance(input_tensors, MMTensor)
    layer_id = mm_data['layer_id']
    mm_data['layer_id'] = layer_id + 1

    # 获取 layer 参数
    '''
    MaxPool2d(int kernel_h=1, int kernel_w=1, int stride_h=1, int stride_w=1, int padding_h=0, int padding_w=0, bool ceil_mode=false);

    '''
    # 创建层
    ceil_mode = torch_layer.ceil_mode
    _ceil_mode = 0
    if ceil_mode:
        _ceil_mode = 1
    layer_arg = 'kernel_h=%d,kernel_w=%d,stride_h=%d,stride_w=%d,padding_h=%d,padding_w=%d,ceil_mode=%d,' % (
        torch_layer.kernel_size, torch_layer.kernel_size,
        torch_layer.stride, torch_layer.stride, torch_layer.padding, torch_layer.padding, _ceil_mode)
    layer = MMLayer(layer_id, 'MaxPool2d', layer_arg)
    # 层添加输入张量
    layer.add_input_tensor(input_tensors)
    # 创建输出张量
    output = create_tensor(mm_data)
    # 层添加输出张量
    layer.add_output_tensor(output)
    # mm_data中添加层
    mm_data['layers'].append(layer)

    return output

# 创建 神经网络层 并且 forward，外部不要调用，而是应该调用 create_layer_and_forward()
def create_AvgPool2d_and_forward(mm_data, torch_layer, input_tensors):
    assert isinstance(input_tensors, MMTensor)
    layer_id = mm_data['layer_id']
    mm_data['layer_id'] = layer_id + 1

    # 获取 layer 参数
    '''
    AvgPool2d(int kernel_h=1, int kernel_w=1, int stride_h=1, int stride_w=1, int padding_h=0, int padding_w=0, bool ceil_mode=false);

    '''
    # 创建层
    ceil_mode = torch_layer.ceil_mode
    _ceil_mode = 0
    if ceil_mode:
        _ceil_mode = 1
    layer_arg = 'kernel_h=%d,kernel_w=%d,stride_h=%d,stride_w=%d,padding_h=%d,padding_w=%d,ceil_mode=%d,' % (
        torch_layer.kernel_size, torch_layer.kernel_size,
        torch_layer.stride, torch_layer.stride, torch_layer.padding, torch_layer.padding, _ceil_mode)
    layer = MMLayer(layer_id, 'AvgPool2d', layer_arg)
    # 层添加输入张量
    layer.add_input_tensor(input_tensors)
    # 创建输出张量
    output = create_tensor(mm_data)
    # 层添加输出张量
    layer.add_output_tensor(output)
    # mm_data中添加层
    mm_data['layers'].append(layer)

    return output


# 创建 神经网络层 并且 forward，外部不要调用，而是应该调用 create_layer_and_forward()
def create_conv2dfuseBN_and_forward(mm_data, torch_layer, bn_layer, input_tensors):
    assert isinstance(input_tensors, MMTensor)
    layer_id = mm_data['layer_id']
    mm_data['layer_id'] = layer_id + 1

    # 获取 layer 参数
    '''
    Conv2d(int in_channels, int out_channels, int kernel_h=1, int kernel_w=1, int stride_h=1, int stride_w=1, int padding_h=0, 
    int padding_w=0, int dilation_h=1, int dilation_w=1, int groups=1, bool use_bias=true, bool create_weights=true);

    '''
    # 创建层
    use_bias = 1
    layer_arg = 'in_channels=%d,out_channels=%d,kernel_h=%d,kernel_w=%d,stride_h=%d,stride_w=%d,padding_h=%d,padding_w=%d,dilation_h=%d,dilation_w=%d,groups=%d,use_bias=%d,' % (
        torch_layer.in_channels, torch_layer.out_channels, torch_layer.kernel_size[0], torch_layer.kernel_size[1],
        torch_layer.stride[0], torch_layer.stride[1], torch_layer.padding[0], torch_layer.padding[1],
        torch_layer.dilation[0], torch_layer.dilation[1], torch_layer.groups, use_bias)
    layer = MMLayer(layer_id, 'Conv2d', layer_arg)
    # 层添加输入张量
    layer.add_input_tensor(input_tensors)
    # 创建输出张量
    output = create_tensor(mm_data)
    # 层添加输出张量
    layer.add_output_tensor(output)
    # mm_data中添加层
    mm_data['layers'].append(layer)

    # 权重写入权重文件
    net_weight_file = mm_data['net_weight_file']


    '''
    合并卷积层和BN层。推导：
    y = [(conv_w * x + conv_b) - bn_m] / sqrt(bn_v + eps) * bn_w + bn_b
    = [conv_w * x + (conv_b - bn_m)] / sqrt(bn_v + eps) * bn_w + bn_b
    = conv_w * x / sqrt(bn_v + eps) * bn_w + (conv_b - bn_m) / sqrt(bn_v + eps) * bn_w + bn_b
    = conv_w * bn_w / sqrt(bn_v + eps) * x + (conv_b - bn_m) / sqrt(bn_v + eps) * bn_w + bn_b

    所以
    new_conv_w = conv_w * bn_w / sqrt(bn_v + eps)
    new_conv_b = (conv_b - bn_m) / sqrt(bn_v + eps) * bn_w + bn_b
    '''

    conv_w = torch_layer.weight.cpu().detach().numpy()
    bn_w = bn_layer.weight.cpu().detach().numpy()
    bn_b = bn_layer.bias.cpu().detach().numpy()
    bn_m = bn_layer.running_mean.cpu().detach().numpy()
    bn_v = bn_layer.running_var.cpu().detach().numpy()
    eps = bn_layer.eps
    if torch_layer.bias is not None:
        conv_b = torch_layer.bias.cpu().detach().numpy()
    else:
        conv_b = np.zeros(bn_w.shape)
    new_conv_w = conv_w * (bn_w / np.sqrt(bn_v + eps)).reshape((-1, 1, 1, 1))
    new_conv_b = (conv_b - bn_m) / np.sqrt(bn_v + eps) * bn_w + bn_b



    # [out_C, in_C, kH, kW] -> [kH, kW, in_C, out_C]
    new_conv_w = new_conv_w.transpose((2, 3, 1, 0))
    # print('conv2d.weight = %s' % get_str(ndarray))

    net_weight_file = bp_write_ndarray(net_weight_file, new_conv_w)
    # print('conv2d.bias = %s' % get_str(ndarray))
    net_weight_file = bp_write_ndarray(net_weight_file, new_conv_b)
    return output


# 创建 神经网络层 并且 forward
def create_layer_and_forward(mm_data, torch_layer, input_tensors, layer_args={}):
    if isinstance(torch_layer, str):
        # 允许用字符串表示神经网络层，比如逐元素相加，relu激活
        # assert torch_layer in ['elementwise_add', 'elementwise_sub', 'elementwise_mul', 'elementwise_div',
        #                        'relu', 'sigmoid', 'silu',
        #                        'concat', 'permute']
        if torch_layer == 'LeakyReLU':  # 导出代码  x = miemienet_utils.create_layer_and_forward(mm_data, 'LeakyReLU', x, {'negative_slope': self.act.negative_slope})
            return create_leakyrelu_and_forward(mm_data, input_tensors, layer_args)
        elif torch_layer == 'Softmax':  # 导出代码  x = miemienet_utils.create_layer_and_forward(mm_data, 'Softmax', x, {'dim': 3})
            return create_Softmax_and_forward(mm_data, input_tensors, layer_args)
        elif torch_layer == 'Transpose':
            # x = x.permute((0, 1, 3, 2))
            # 导出代码  x = miemienet_utils.create_layer_and_forward(mm_data, 'Transpose', x, {'trans_type': '0213'})
            return create_Transpose_and_forward(mm_data, input_tensors, layer_args)
        elif torch_layer == 'Reshape':
            #  python代码     x = x.reshape([1, 4, 8, 36864])
            # 导出代码  x = miemienet_utils.create_layer_and_forward(mm_data, 'Reshape', x, {'dim0': 1, 'dim1': 36864, 'dim2': 4, 'dim3': 8})
            return create_Reshape_and_forward(mm_data, input_tensors, layer_args)
        elif torch_layer == 'AvgPool2d':
            # 全局平均池化， python代码 x = x.mean((2, 3), keepdim=True)
            # 或者是                  x = F.adaptive_avg_pool2d(x, (1, 1))
            # 导出代码  x = miemienet_utils.create_layer_and_forward(mm_data, 'AvgPool2d', x, {'globelpool': 1})
            return create_AvgPool2d_str_and_forward(mm_data, input_tensors, layer_args)
        elif torch_layer == 'ActNoArgs':
            # 无参的激活函数。导出代码 举例：
            # x = miemienet_utils.create_layer_and_forward(mm_data, 'ActNoArgs', x, {'act_type': 'ReLU'})
            # x = miemienet_utils.create_layer_and_forward(mm_data, 'ActNoArgs', x, {'act_type': 'Sigmoid'})
            # x = miemienet_utils.create_layer_and_forward(mm_data, 'ActNoArgs', x, {'act_type': 'SiLU'})
            # x = miemienet_utils.create_layer_and_forward(mm_data, 'ActNoArgs', x, {'act_type': 'Hardsigmoid'})
            return create_act_no_args_and_forward(mm_data, input_tensors, layer_args)
        elif torch_layer == 'Interp':
            # 插值。 举例：
            # x = F.interpolate(x, scale_factor=2.)   对应
            # x = miemienet_utils.create_layer_and_forward(mm_data, 'Interp', x, {'size_h': 0, 'size_w': 0, 'scale_h': 2., 'scale_w': 2., 'mode': 'nearest', 'align_corners': 0, 'recompute_scale_factor': 0})
            return create_Interp_and_forward(mm_data, input_tensors, layer_args)
        elif torch_layer == 'ElementWise':
            # 导出代码
            # x = miemienet_utils.create_layer_and_forward(mm_data, 'ElementWise', [x, res], {'op_type': 'add'})
            # x = miemienet_utils.create_layer_and_forward(mm_data, 'ElementWise', [x, res], {'op_type': 'sub'})
            # x = miemienet_utils.create_layer_and_forward(mm_data, 'ElementWise', [x, res], {'op_type': 'mul'})
            # x = miemienet_utils.create_layer_and_forward(mm_data, 'ElementWise', [x, res], {'op_type': 'div'})
            return create_ElementWise_and_forward(mm_data, input_tensors, layer_args)
        elif torch_layer == 'ConstAdd':  # 一个张量和一个常数相加，比如x = x + 3.14,  导出代码  x = miemienet_utils.create_layer_and_forward(mm_data, 'ConstAdd', x, {'value': 3.14})
            return create_Constxxx_and_forward(mm_data, input_tensors, layer_args, 'Add')
        elif torch_layer == 'ConstSub':  # 一个张量和一个常数相减，比如x = x - 3.14,  导出代码  x = miemienet_utils.create_layer_and_forward(mm_data, 'ConstSub', x, {'value': 3.14})
            return create_Constxxx_and_forward(mm_data, input_tensors, layer_args, 'Sub')
        elif torch_layer == 'ConstMul':  # 一个张量和一个常数相乘，比如x = x * 3.14,  导出代码  x = miemienet_utils.create_layer_and_forward(mm_data, 'ConstMul', x, {'value': 3.14})
            return create_Constxxx_and_forward(mm_data, input_tensors, layer_args, 'Mul')
        elif torch_layer == 'ConstDiv':  # 一个张量和一个常数相除，比如x = x / 3.14,  导出代码  x = miemienet_utils.create_layer_and_forward(mm_data, 'ConstDiv', x, {'value': 3.14})
            return create_Constxxx_and_forward(mm_data, input_tensors, layer_args, 'Div')
        elif torch_layer == 'ConCat':
            # python代码 x = torch.cat([x, res], 1),  导出代码  x = miemienet_utils.create_layer_and_forward(mm_data, 'ConCat', [x, res], {'dim': 3})
            # 注意！！！这是图像任务时的做法，此时 x.shape = NCHW，但是miemienet的特征图排列是 NHWC，要灵活变通。至于其他维度的张量，先不管，遇到再说
            return create_ConCat_and_forward(mm_data, input_tensors, layer_args)
        else:
            raise NotImplementedError("not implemented.")
    else:
        if isinstance(torch_layer, nn.Conv2d):  # 导出代码  x = miemienet_utils.create_layer_and_forward(mm_data, self.conv, x)
            return create_conv2d_and_forward(mm_data, torch_layer, input_tensors)
        elif isinstance(torch_layer, list):  # 多层fuse成1层
            if isinstance(torch_layer[0], nn.Conv2d) and isinstance(torch_layer[1], nn.BatchNorm2d):
                # 卷积和BN融合, python代码 res = self.conv2(x); res = self.bn(res)
                # 导出代码  res = miemienet_utils.create_layer_and_forward(mm_data, [self.conv2, self.bn], x)
                return create_conv2dfuseBN_and_forward(mm_data, torch_layer[0], torch_layer[1], input_tensors)
            else:
                raise NotImplementedError("not implemented.")
        elif isinstance(torch_layer, nn.MaxPool2d):  # 导出代码  x = miemienet_utils.create_layer_and_forward(mm_data, self.pool, x)
            return create_MaxPool2d_and_forward(mm_data, torch_layer, input_tensors)
        elif isinstance(torch_layer, nn.AvgPool2d):  # 导出代码  x = miemienet_utils.create_layer_and_forward(mm_data, self.pool, x)
            return create_AvgPool2d_and_forward(mm_data, torch_layer, input_tensors)
        else:
            raise NotImplementedError("not implemented.")


def create_new_param_bin(save_name, input_num):
    net_weight_file = open('%s.bin' % save_name, 'wb')
    net_content = ''
    layer_id = 0
    tensor_id = 0

    mm_data = {}
    mm_data['net_weight_file'] = net_weight_file
    mm_data['net_content'] = net_content
    mm_data['layer_id'] = layer_id
    mm_data['tensor_id'] = tensor_id
    mm_data['layers'] = []
    mm_data['save_name'] = save_name
    return mm_data


def save_miemienet_model(save_name, ncnn_data, bottom_names, replace_input_names=[], replace_output_names=[]):
    assert isinstance(bottom_names, list)
    assert isinstance(replace_input_names, list)
    assert isinstance(replace_output_names, list)
    assert len(bottom_names) == len(replace_output_names)
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']
    for i in range(len(replace_input_names)):
        pp = pp.replace('tensor_%.8d' % (i,), replace_input_names[i])
    for i in range(len(replace_output_names)):
        pp = pp.replace(bottom_names[i], replace_output_names[i])
    pp = '7767517\n%d %d\n' % (layer_id, tensor_id) + pp
    with open('%s.param' % save_name, 'w', encoding='utf-8') as f:
        f.write(pp)
        f.close()
    return ncnn_data, bottom_names


def save_as_bin(name, ndarray):
    assert isinstance(ndarray, np.ndarray)
    _file = open(name, 'wb')
    _file = bp_write_ndarray(_file, ndarray)


def save_as_txt(name, ndarray):
    assert isinstance(ndarray, np.ndarray)
    save_ndarray_as_txt(name, ndarray)


def save_ndarray_as_txt(name, ndarray):
    content = ''
    array_flatten = np.copy(ndarray)
    array_flatten = np.reshape(array_flatten, (-1, ))
    n = array_flatten.shape[0]
    for i in range(n):
        content += '%f\n' % array_flatten[i]
    with open(name, 'w', encoding='utf-8') as f:
        f.write(content)
        f.close()


