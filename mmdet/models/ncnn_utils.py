#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description : 大部分参考了onnx2ncnn.cpp的代码
#
# ================================================================
import struct
import numpy as np


def newest_bottom_names(ncnn_data):
    tensor_id = ncnn_data['tensor_id']
    bottom_names = ['tensor_%.8d' % (tensor_id - 1,), ]
    return bottom_names


def create_top_names(ncnn_data, num):
    assert num >= 1
    tensor_id = ncnn_data['tensor_id']
    # if tensor_id == 242:
    #     print()
    top_names = []
    for i in range(num):
        top_names.append('tensor_%.8d' % (tensor_id + i,))
    return top_names


def rename_tensor(ncnn_data, bottom_names):
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    tensor_id = 0
    lines = pp.split('\n')
    lines = lines[:-1]

    tensors_dic = {}
    tensor_id = 0
    for i, line in enumerate(lines):
        ss = line.split()
        in_num = int(ss[2])
        out_num = int(ss[3])
        p = 4
        for i1 in range(in_num):
            tensor_name = ss[p]
            if tensor_name not in tensors_dic.keys():
                aaaaaaaaaa = 'tensor_%.8d' % (tensor_id, )
                tensor_id += 1
                tensors_dic[tensor_name] = aaaaaaaaaa
            p += 1
        for i2 in range(out_num):
            tensor_name = ss[p]
            if tensor_name not in tensors_dic.keys():
                aaaaaaaaaa = 'tensor_%.8d' % (tensor_id, )
                tensor_id += 1
                tensors_dic[tensor_name] = aaaaaaaaaa
            p += 1
    content = ''
    for i, line in enumerate(lines):
        ss = line.split()
        in_num = int(ss[2])
        out_num = int(ss[3])
        p = 4 + in_num + out_num - 1
        for i1 in range(in_num):
            tensor_name = ss[p]
            ss[p] = tensors_dic[tensor_name]
            p -= 1
        for i2 in range(out_num):
            tensor_name = ss[p]
            ss[p] = tensors_dic[tensor_name]
            p -= 1
        line2 = ''
        for kkk, s in enumerate(ss):
            if kkk == 0:
                line2 += s
            elif kkk == 1:
                line2 += '\t' + s
            elif kkk == 2:
                line2 += '\t' + s
            else:
                line2 += ' ' + s
        content += line2 + '\n'
    pp = content

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    new_bottom_names = []
    for bname in bottom_names:
        new_bottom_names.append(tensors_dic[bname])
    return new_bottom_names


def split_input_tensor(ncnn_data, bottom_names):
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    lines = pp.split('\n')
    lines = lines[:-1]

    # 统计张量被作为输入的次数
    tensor_as_input_count = {}
    for i, line in enumerate(lines):
        ss = line.split()
        in_num = int(ss[2])
        out_num = int(ss[3])
        p = 4
        for i1 in range(in_num):
            tensor_name = ss[p]
            if tensor_name not in tensor_as_input_count.keys():
                tensor_as_input_count[tensor_name] = 1
            else:
                tensor_as_input_count[tensor_name] += 1
            p += 1


    keys = tensor_as_input_count.keys()
    for split_tensor_name in keys:
        count = tensor_as_input_count[split_tensor_name]
        if count > 1:
            # 给网络插入1个Split层
            new_lines = []
            # 找到输出首次是split_tensor_name的层，在这个层的后面插入Split层
            find = False
            copy_i = 0
            for i, line in enumerate(lines):
                if not find:
                    ss = line.split()
                    in_num = int(ss[2])
                    out_num = int(ss[3])
                    p = 4 + in_num
                    for i2 in range(out_num):
                        tensor_name = ss[p]
                        if tensor_name == split_tensor_name:
                            find = True
                        p += 1
                    new_lines.append(line)
                    if find:
                        # 马上在当前层的后面插入Split层
                        layer = 'Split\tlayer_%.8d\t1 %d %s' % (layer_id, count, split_tensor_name)
                        for ii in range(count):
                            layer += ' %s_%.6d' % (split_tensor_name, ii)
                        layer_id += 1
                        tensor_id += count
                        new_lines.append(layer)
                else:
                    # 输入张量是split_tensor_name的层，替换成Split层的每一个输出。
                    ss = line.split()
                    in_num = int(ss[2])
                    out_num = int(ss[3])
                    p = 4
                    for i1 in range(in_num):
                        tensor_name = ss[p]
                        if tensor_name == split_tensor_name:
                            ss[p] = '%s_%.6d' % (split_tensor_name, copy_i)
                            copy_i += 1
                        p += 1
                    line2 = ''
                    for kkk, s in enumerate(ss):
                        if kkk == 0:
                            line2 += s
                        elif kkk == 1:
                            line2 += '\t' + s
                        elif kkk == 2:
                            line2 += '\t' + s
                        else:
                            line2 += ' ' + s
                    new_lines.append(line2)
            lines = new_lines

    pp = ''
    for i, line in enumerate(lines):
        pp += line + '\n'
    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    bottom_names = rename_tensor(ncnn_data, bottom_names)
    return bottom_names


def conv2d(ncnn_data, bottom_names, conv):
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    top_names = create_top_names(ncnn_data, num=1)
    pp += 'Convolution\tlayer_%.8d\t1 1 %s %s' % (layer_id, bottom_names[0], top_names[0])
    pp += ' 0=%d' % conv.out_channels
    if len(conv.kernel_size) == 2:
        pp += ' 1=%d' % conv.kernel_size[1]
        pp += ' 11=%d' % conv.kernel_size[0]
    else:
        raise NotImplementedError("not implemented.")
    if len(conv.stride) == 2:
        pp += ' 3=%d' % conv.stride[1]
        pp += ' 13=%d' % conv.stride[0]
    else:
        raise NotImplementedError("not implemented.")
    if len(conv.padding) == 2:
        pp += ' 4=%d' % conv.padding[1]
        pp += ' 14=%d' % conv.padding[0]
    else:
        raise NotImplementedError("not implemented.")
    if conv.bias is not None:
        pp += ' 5=1'
    else:
        pp += ' 5=0'
    out_C, in_C, kH, kW = conv.weight.shape
    w_ele_num = out_C * in_C * kH * kW
    pp += ' 6=%d' % w_ele_num
    assert conv.groups == 1
    if conv.groups > 1:
        pp += ' 7=%d' % conv.groups
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    # 卷积层写入权重。参考了onnx2ncnn，开头写个0
    s = struct.pack('i', 0)
    bp.write(s)

    conv_w = conv.weight.cpu().detach().numpy()
    for i1 in range(out_C):
        for i2 in range(in_C):
            for i3 in range(kH):
                for i4 in range(kW):
                    s = struct.pack('f', conv_w[i1][i2][i3][i4])
                    bp.write(s)
    if conv.bias is not None:
        conv_b = conv.bias.cpu().detach().numpy()
        for i1 in range(out_C):
            s = struct.pack('f', conv_b[i1])
            bp.write(s)
    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def fuse_conv_bn(ncnn_data, bottom_names, conv, bn):
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    top_names = create_top_names(ncnn_data, num=1)
    pp += 'Convolution\tlayer_%.8d\t1 1 %s %s' % (layer_id, bottom_names[0], top_names[0])
    pp += ' 0=%d' % conv.out_channels
    if len(conv.kernel_size) == 2:
        pp += ' 1=%d' % conv.kernel_size[1]
        pp += ' 11=%d' % conv.kernel_size[0]
    else:
        raise NotImplementedError("not implemented.")
    if len(conv.stride) == 2:
        pp += ' 3=%d' % conv.stride[1]
        pp += ' 13=%d' % conv.stride[0]
    else:
        raise NotImplementedError("not implemented.")
    if len(conv.padding) == 2:
        pp += ' 4=%d' % conv.padding[1]
        pp += ' 14=%d' % conv.padding[0]
    else:
        raise NotImplementedError("not implemented.")
    # 合并卷积层和BN层。肯定使用了偏移bias
    pp += ' 5=1'
    out_C, in_C, kH, kW = conv.weight.shape
    w_ele_num = out_C * in_C * kH * kW
    pp += ' 6=%d' % w_ele_num
    assert conv.groups == 1
    if conv.groups > 1:
        pp += ' 7=%d' % conv.groups
    pp += '\n'
    layer_id += 1
    tensor_id += 1

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

    # 卷积层写入权重。参考了onnx2ncnn，开头写个0
    s = struct.pack('i', 0)
    bp.write(s)

    conv_w = conv.weight.cpu().detach().numpy()
    bn_w = bn.weight.cpu().detach().numpy()
    bn_b = bn.bias.cpu().detach().numpy()
    bn_m = bn.running_mean.cpu().detach().numpy()
    bn_v = bn.running_var.cpu().detach().numpy()
    eps = bn.eps
    if conv.bias is not None:
        conv_b = conv.bias.cpu().detach().numpy()
    else:
        conv_b = np.zeros(bn_w.shape)
    new_conv_w = conv_w * (bn_w / np.sqrt(bn_v + eps)).reshape((-1, 1, 1, 1))
    new_conv_b = (conv_b - bn_m) / np.sqrt(bn_v + eps) * bn_w + bn_b
    for i1 in range(out_C):
        for i2 in range(in_C):
            for i3 in range(kH):
                for i4 in range(kW):
                    s = struct.pack('f', new_conv_w[i1][i2][i3][i4])
                    bp.write(s)
    for i1 in range(out_C):
        s = struct.pack('f', new_conv_b[i1])
        bp.write(s)
    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def pooling(ncnn_data, bottom_names, op, pool):
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    op_id = -1
    if op == 'MaxPool':
        op_id = 0
    elif op == 'AveragePool':
        op_id = 1
    else:
        raise NotImplementedError("not implemented.")
    dilation = pool.dilation
    assert dilation == 1
    ceil_mode = pool.ceil_mode
    pad_mode = -1
    if ceil_mode:
        pad_mode = 0
    else:
        pad_mode = 1

    top_names = create_top_names(ncnn_data, num=1)
    pp += 'Pooling\tlayer_%.8d\t1 1 %s %s' % (layer_id, bottom_names[0], top_names[0])
    pp += ' 0=%d' % op_id
    if isinstance(pool.kernel_size, int):
        pp += ' 1=%d' % pool.kernel_size
    elif isinstance(pool.kernel_size, (tuple, list)) and len(pool.kernel_size) == 2:
        pp += ' 1=%d' % pool.kernel_size[1]
        pp += ' 11=%d' % pool.kernel_size[0]
    else:
        raise NotImplementedError("not implemented.")
    if isinstance(pool.stride, int):
        pp += ' 2=%d' % pool.stride
    elif isinstance(pool.stride, (tuple, list)) and len(pool.stride) == 2:
        pp += ' 2=%d' % pool.stride[1]
        pp += ' 12=%d' % pool.stride[0]
    else:
        raise NotImplementedError("not implemented.")
    if isinstance(pool.padding, int):
        pp += ' 3=%d' % pool.padding
    elif isinstance(pool.padding, (tuple, list)) and len(pool.padding) == 2:
        pp += ' 3=%d' % pool.padding[1]
        pp += ' 13=%d' % pool.padding[0]
    else:
        raise NotImplementedError("not implemented.")
    pp += ' 5=%d' % pad_mode
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def activation(ncnn_data, bottom_names, act_name, args={}):
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    top_names = create_top_names(ncnn_data, num=1)
    if act_name == None:
        top_names = bottom_names
    elif act_name == 'swish':
        pp += 'Swish\tlayer_%.8d\t1 1 %s %s\n' % (layer_id, bottom_names[0], top_names[0])
        layer_id += 1
        tensor_id += 1
    elif act_name == 'sigmoid':
        pp += 'Sigmoid\tlayer_%.8d\t1 1 %s %s\n' % (layer_id, bottom_names[0], top_names[0])
        layer_id += 1
        tensor_id += 1
    elif act_name == 'mish':
        pp += 'Mish\tlayer_%.8d\t1 1 %s %s\n' % (layer_id, bottom_names[0], top_names[0])
        layer_id += 1
        tensor_id += 1
    elif act_name == 'hardsigmoid':
        pp += 'HardSigmoid\tlayer_%.8d\t1 1 %s %s 0=1.666667e-01 1=5.000000e-01\n' % (layer_id, bottom_names[0], top_names[0])
        layer_id += 1
        tensor_id += 1
    elif act_name == 'leaky_relu':
        negative_slope = args['negative_slope']
        assert isinstance(negative_slope, float)
        pp += 'ReLU\tlayer_%.8d\t1 1 %s %s 0=%e\n' % (layer_id, bottom_names[0], top_names[0], negative_slope)
        layer_id += 1
        tensor_id += 1
    elif act_name == 'relu':
        pp += 'ReLU\tlayer_%.8d\t1 1 %s %s 0=0.0\n' % (layer_id, bottom_names[0], top_names[0])
        layer_id += 1
        tensor_id += 1
    else:
        raise NotImplementedError("not implemented.")

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def split(ncnn_data, bottom_names, num):
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    top_names = create_top_names(ncnn_data, num=num)
    pp += 'Split\tlayer_%.8d\t1 %d %s' % (layer_id, num, bottom_names[0])
    for i in range(num):
        pp += ' %s' % top_names[i]
    pp += '\n'
    layer_id += 1
    tensor_id += num

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def binaryOp(ncnn_data, bottom_names, op):
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    top_names = create_top_names(ncnn_data, num=1)
    op_id = -1
    if op == 'Add':
        op_id = 0
    elif op == 'Sub':
        op_id = 1
    elif op == 'Mul':
        op_id = 2
    elif op == 'Div':
        op_id = 3
    elif op == 'Max':
        op_id = 4
    elif op == 'Min':
        op_id = 5
    elif op == 'Pow':
        op_id = 6
    elif op == 'RSub':
        op_id = 7
    elif op == 'RDiv':
        op_id = 8
    else:
        raise NotImplementedError("not implemented.")

    num = len(bottom_names)
    pp += 'BinaryOp\tlayer_%.8d\t%d 1' % (layer_id, num)
    for i in range(num):
        pp += ' %s' % bottom_names[i]
    pp += ' %s' % top_names[0]
    pp += ' 0=%d' % op_id
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def reduction(ncnn_data, bottom_names, op, input_dims, dims, keepdim=False):
    assert isinstance(dims, (list, tuple))
    assert len(dims) > 0
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    '''
    见
    int Reduction::load_param(const ParamDict& pd)
        {
            operation = pd.get(0, 0);
            reduce_all = pd.get(1, 1);
            coeff = pd.get(2, 1.f);
            axes = pd.get(3, Mat());
            keepdims = pd.get(4, 0);

            return 0;
        }
    0=3表示mean


    onnx2ncnn.cpp
        else if (op == "ReduceMax" || op == "ReduceMin" || op == "ReduceMean" || op == "ReduceProd" || op == "ReduceSum" || op == "ReduceSumSquare" || op == "ReduceL1" || op == "ReduceL2" || op == "ReduceLogSum" || op == "ReduceLogSumExp")
        {
            int op_type = -233;
            if (op == "ReduceSum")
                op_type = 0;
            else if (op == "ReduceSumSquare")
                op_type = 2;
            else if (op == "ReduceMean")
                op_type = 3;
            else if (op == "ReduceMax")
                op_type = 4;
            else if (op == "ReduceMin")
                op_type = 5;
            else if (op == "ReduceProd")
                op_type = 6;
            else if (op == "ReduceL1")
                op_type = 7;
            else if (op == "ReduceL2")
                op_type = 8;
            else if (op == "ReduceLogSum")
                op_type = 9;
            else if (op == "ReduceLogSumExp")
                op_type = 10;
            fprintf(pp, " 0=%d", op_type);

            std::vector<int> axes = get_node_attr_ai(node, "axes");
            int keepdims = get_node_attr_i(node, "keepdims", 1);
    '''

    top_names = create_top_names(ncnn_data, num=1)
    op_id = -1
    if op == 'ReduceSum':
        op_id = 0
    elif op == 'ReduceSumSquare':
        op_id = 2
    elif op == 'ReduceMean':
        op_id = 3
    elif op == 'ReduceMax':
        op_id = 4
    elif op == 'ReduceMin':
        op_id = 5
    elif op == 'ReduceProd':
        op_id = 6
    elif op == 'ReduceL1':
        op_id = 7
    elif op == 'ReduceL2':
        op_id = 8
    elif op == 'ReduceLogSum':
        op_id = 9
    elif op == 'ReduceLogSumExp':
        op_id = 10
    else:
        raise NotImplementedError("not implemented.")

    pp += 'Reduction\tlayer_%.8d\t1 1 %s %s 0=%d' % (layer_id, bottom_names[0], top_names[0], op_id)
    reduce_all = False
    if input_dims == len(dims):
        reduce_all = True
    if reduce_all:
        pp += ' 1=1'
    else:
        pp += ' 1=0'
    # 被干掉的维的信息。先填dims的长度，再填每个dim。由于ncnn中处理图片时是三维张量，所以填入dim-1
    pp += ' -23303=%d' % (len(dims),)
    for dim in dims:
        pp += ',%d' % (dim - 1,)
    if keepdim:
        pp += ' 4=1'
    else:
        pp += ' 4=0'
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def crop(ncnn_data, bottom_names, starts, ends, axes):
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    '''
        x = x[:, :2, :4, :4] 可以被翻译成
Crop             layer_name                  1 1 tensor_xx tensor_xxx -23309=3,0,0,0 -23310=3,2,4,4 -23311=3,0,1,2
        x = x[:, :2, :, :] 可以被翻译成
Crop             layer_name                  1 1 tensor_xx tensor_xxx -23309=1,0 -23310=1,2 -23311=1,0
即：
-23309第0个参数表示有几个维做切片，第1个参数表示第0个做切片的维开始切片的下标，第2个参数表示第1个做切片的维开始切片的下标，...
-23310第0个参数表示有几个维做切片，第1个参数表示第0个做切片的维结束切片的下标，第2个参数表示第1个做切片的维结束切片的下标，...
-23311第0个参数表示有几个维做切片，第1个参数表示第0个做切片的维的下标，第2个参数表示第1个做切片的维的下标，...

由于ncnn中处理图片时是三维张量，所以填入的维的下标需要-1
    '''

    top_names = create_top_names(ncnn_data, num=1)
    pp += 'Crop\tlayer_%.8d\t1 1 %s %s' % (layer_id, bottom_names[0], top_names[0])
    pp += ' -23309=%s -23310=%s -23311=%s' % (starts, ends, axes)
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def concat(ncnn_data, bottom_names, dim):
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    '''
Concat           Concat_33                2 1 67 83 84 0=0
    '''

    top_names = create_top_names(ncnn_data, num=1)
    num = len(bottom_names)
    pp += 'Concat\tlayer_%.8d\t%d 1' % (layer_id, num)
    for i in range(num):
        pp += ' %s' % bottom_names[i]
    pp += ' %s' % top_names[0]
    pp += ' 0=%d' % (dim - 1, )
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def interpolate(ncnn_data, bottom_names, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    assert size is None
    output_height = 0
    output_width = 0
    assert isinstance(scale_factor, float)
    top_names = create_top_names(ncnn_data, num=1)
    pp += 'Interp\tlayer_%.8d\t1 1 %s %s' % (layer_id, bottom_names[0], top_names[0])


    resize_type = 1
    if mode == 'nearest':
        resize_type = 1
    elif mode == 'bilinear':
        resize_type = 2
    elif mode == 'bicubic':
        resize_type = 3
    else:
        raise NotImplementedError("not implemented.")
    align_corner = 0
    if align_corners:
        align_corner = 1
    pp += ' 0=%d 1=%e 2=%e' % (resize_type, scale_factor, scale_factor)
    pp += ' 3=%d 4=%d 6=%d\n' % (output_height, output_width, align_corner)
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def permute(ncnn_data, bottom_names, perm=None):
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    # 死记硬背
    args = ''
    if perm == '(0, 2, 3, 1)':
        args = ' 0=3'
    elif perm == '(0, 3, 1, 2)':
        args = ' 0=4'
    elif perm == '(0, 2, 1, 3)':
        args = ' 0=2'
    elif perm == '(0, 2, 1)':
        args = ' 0=1'
    else:
        raise NotImplementedError("not implemented.")

    top_names = create_top_names(ncnn_data, num=1)
    pp += 'Permute\tlayer_%.8d\t1 1 %s %s' % (layer_id, bottom_names[0], top_names[0])
    pp += args
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def adaptive_avg_pool2d(ncnn_data, bottom_names, output_size=None):
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    # 死记硬背.全局平均池化
    args = ''
    if output_size == '(1, 1)':
        args = ' 0=1 4=1'
    else:
        raise NotImplementedError("not implemented.")

    top_names = create_top_names(ncnn_data, num=1)
    pp += 'Pooling\tlayer_%.8d\t1 1 %s %s' % (layer_id, bottom_names[0], top_names[0])
    pp += args
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def reshape(ncnn_data, bottom_names, shape):
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    #
    args = ''
    if len(shape) == 1:
        args = ' 0=%d'%(shape[0], )
    elif len(shape) == 2:
        args = ' 0=%d'%(shape[1], )
    elif len(shape) == 3:
        args = ' 0=%d 1=%d'%(shape[2], shape[1])
    elif len(shape) == 4:
        args = ' 0=%d 1=%d 2=%d'%(shape[3], shape[2], shape[1])
    elif len(shape) == 5:
        args = ' 0=%d 1=%d 2=%d'%(shape[4] * shape[3], shape[2], shape[1])
    else:
        raise NotImplementedError("not implemented.")

    top_names = create_top_names(ncnn_data, num=1)
    pp += 'Reshape\tlayer_%.8d\t1 1 %s %s' % (layer_id, bottom_names[0], top_names[0])
    pp += args
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def softmax(ncnn_data, bottom_names, dim):
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    assert isinstance(dim, int)
    assert dim >= 0
    args = ' 0=%d 1=1'%(dim - 1, )

    top_names = create_top_names(ncnn_data, num=1)
    pp += 'Softmax\tlayer_%.8d\t1 1 %s %s' % (layer_id, bottom_names[0], top_names[0])
    pp += args
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names




