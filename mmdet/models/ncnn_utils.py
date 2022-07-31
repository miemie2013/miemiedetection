#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description : 参考了部分onnx2ncnn.cpp的代码
#
# ================================================================
import struct
import numpy as np


def create_new_param_bin(save_name, input_num):
    bp = open('%s.bin' % save_name, 'wb')
    pp = ''
    layer_id = 0
    tensor_id = 0
    bottom_names = []
    pp += 'Input\tlayer_%.8d\t0 %d' % (layer_id, input_num)
    for i in range(input_num):
        pp += ' tensor_%.8d' % tensor_id
        bottom_names.append('tensor_%.8d' % tensor_id)
        tensor_id += 1
    pp += '\n'
    layer_id += 1

    ncnn_data = {}
    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return ncnn_data, bottom_names


def save_param(save_name, ncnn_data, bottom_names, replace_input_names=[], replace_output_names=[]):
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


def newest_bottom_names(ncnn_data):
    tensor_id = ncnn_data['tensor_id']
    bottom_names = ['tensor_%.8d' % (tensor_id - 1,), ]
    return bottom_names


def check_bottom_names(bottom_names):
    if isinstance(bottom_names, str):
        bottom_names = [bottom_names, ]
    elif isinstance(bottom_names, list):
        all_is_str = True
        num_input = len(bottom_names)
        for i in range(num_input):
            if not isinstance(bottom_names[i], str):
                all_is_str = False
                break
        if not all_is_str:
            raise NotImplementedError("bottom_names elements type not implemented.")
    else:
        raise NotImplementedError("bottom_names type not implemented.")
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


def pretty_format(ncnn_data, bottom_names):
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    lines = pp.split('\n')
    lines = lines[:-1]
    content = ''
    for i, line in enumerate(lines):
        ss = line.split()
        line2 = ''
        for kkk, s in enumerate(ss):
            if kkk == 0:
                line2 += "%-24s"%s
            elif kkk == 1:
                line2 += ' %-24s'%s
            elif kkk == 2:
                line2 += ' ' + s
            else:
                line2 += ' ' + s
        content += line2 + '\n'
    pp = content

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return bottom_names


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
    bottom_names = check_bottom_names(bottom_names)
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
    bottom_names = pretty_format(ncnn_data, bottom_names)
    return bottom_names


def conv2d(ncnn_data, bottom_names, conv, act_name=None, act_param_dict=None):
    bottom_names = check_bottom_names(bottom_names)
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
    # 合并激活
    pp = fused_activation(pp, act_name, act_param_dict)
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


def deformable_conv2d(ncnn_data, bottom_names, conv, act_name=None, act_param_dict=None):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    top_names = create_top_names(ncnn_data, num=1)
    num = len(bottom_names)
    pp += 'DeformableConv2D\tlayer_%.8d\t%d 1' % (layer_id, num)
    for i in range(num):
        pp += ' %s' % bottom_names[i]
    pp += ' %s' % top_names[0]
    pp += ' 0=%d' % conv.out_channels
    if len(conv.kernel_size) == 2:
        pp += ' 1=%d' % conv.kernel_size[1]
        pp += ' 11=%d' % conv.kernel_size[0]
    else:
        raise NotImplementedError("not implemented.")
    if len(conv.stride) == 2:
        pp += ' 3=%d' % conv.stride[1]
        pp += ' 13=%d' % conv.stride[0]
    elif len(conv.stride) == 1:
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
    # 合并激活
    pp = fused_activation(pp, act_name, act_param_dict)
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


def PPYOLODecode(ncnn_data, bottom_names, num_classes, anchors, anchor_masks, strides, scale_x_y=1., iou_aware_factor=0.5, obj_thr=0.1, anchor_per_stride=3):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    top_names = create_top_names(ncnn_data, num=2)
    num = len(bottom_names)
    pp += 'PPYOLODecode\tlayer_%.8d\t%d 2' % (layer_id, num)
    for i in range(num):
        pp += ' %s' % bottom_names[i]
    pp += ' %s' % top_names[0]
    pp += ' %s' % top_names[1]
    pp += ' 0=%d' % num_classes

    pp += ' -23301=%d' % (anchors.shape[0] * anchors.shape[1], )
    for i in range(len(strides)):
        anchors_this_stride = anchors[anchor_masks[i]]
        anchors_this_stride = np.reshape(anchors_this_stride, (-1, ))
        for ele in anchors_this_stride:
            pp += ',%e' % (ele, )

    pp += ' -23302=%d' % (len(strides), )
    for i in range(len(strides)):
        stride = strides[i]
        pp += ',%e' % (stride, )

    pp += ' 3=%e' % scale_x_y
    pp += ' 4=%e' % iou_aware_factor
    pp += ' 5=%e' % obj_thr
    pp += ' 6=%d' % anchor_per_stride
    pp += '\n'
    layer_id += 1
    tensor_id += 2

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def support_fused_activation(act_name):
    # print(act_name)
    # 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid 5=mish 6=hardswish
    support = False
    if act_name == None:
        support = False
    elif act_name == 'relu':
        support = True
    elif act_name == 'leaky_relu':
        support = True
    elif act_name == 'clip':
        support = True
    elif act_name == 'sigmoid':
        support = True
    elif act_name == 'mish':
        support = True
    elif act_name == 'hardswish':
        support = True
    return support


def fused_activation(pp, act_name, act_param_dict):
    # print(act_name)
    # 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid 5=mish 6=hardswish
    if act_name == None:
        pass
    elif act_name == 'relu':
        pp += ' 9=1'
    elif act_name == 'leaky_relu':
        pp += ' 9=2'
        negative_slope = act_param_dict['negative_slope']
        assert isinstance(negative_slope, float)
        pp += ' -23310=1,%e' % negative_slope
    elif act_name == 'clip':
        # 比如卷积层后接一句 x = x.clamp(-55.8, 0.89)    pnnx转换后带 -23310=2,-5.580000e+01,8.900000e-01
        pp += ' 9=3'
        min_v = act_param_dict['min_v']
        assert isinstance(min_v, float)
        max_v = act_param_dict['max_v']
        assert isinstance(max_v, float)
        pp += ' -23310=2,%e,%e' % (min_v, max_v)
    elif act_name == 'sigmoid':
        pp += ' 9=4'
    elif act_name == 'mish':
        pp += ' 9=5'
    elif act_name == 'hardswish':
        pp += ' 9=6'
        alpha = act_param_dict['alpha']
        assert isinstance(alpha, float)
        beta = act_param_dict['beta']
        assert isinstance(beta, float)
        pp += ' -23310=2,%e,%e' % (alpha, beta)
    else:
        raise NotImplementedError("not implemented.")
    return pp


def fuse_conv_bn(ncnn_data, bottom_names, conv, bn, act_name=None, act_param_dict=None):
    bottom_names = check_bottom_names(bottom_names)
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
    # 合并激活
    pp = fused_activation(pp, act_name, act_param_dict)
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


def fuse_deformconv_bn(ncnn_data, bottom_names, conv, bn, act_name=None, act_param_dict=None):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    top_names = create_top_names(ncnn_data, num=1)
    num = len(bottom_names)
    pp += 'DeformableConv2D\tlayer_%.8d\t%d 1' % (layer_id, num)
    for i in range(num):
        pp += ' %s' % bottom_names[i]
    pp += ' %s' % top_names[0]
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
    # 合并激活
    pp = fused_activation(pp, act_name, act_param_dict)
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
    bottom_names = check_bottom_names(bottom_names)
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


def Fpooling(ncnn_data, bottom_names, op, kernel_size, stride, padding=0, dilation=1, ceil_mode=False):
    bottom_names = check_bottom_names(bottom_names)
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
    assert dilation == 1
    pad_mode = -1
    if ceil_mode:
        pad_mode = 0
    else:
        pad_mode = 1

    top_names = create_top_names(ncnn_data, num=1)
    pp += 'Pooling\tlayer_%.8d\t1 1 %s %s' % (layer_id, bottom_names[0], top_names[0])
    pp += ' 0=%d' % op_id
    if isinstance(kernel_size, int):
        pp += ' 1=%d' % kernel_size
    elif isinstance(kernel_size, (tuple, list)) and len(kernel_size) == 2:
        pp += ' 1=%d' % kernel_size[1]
        pp += ' 11=%d' % kernel_size[0]
    else:
        raise NotImplementedError("not implemented.")
    if isinstance(stride, int):
        pp += ' 2=%d' % stride
    elif isinstance(stride, (tuple, list)) and len(stride) == 2:
        pp += ' 2=%d' % stride[1]
        pp += ' 12=%d' % stride[0]
    else:
        raise NotImplementedError("not implemented.")
    if isinstance(padding, int):
        pp += ' 3=%d' % padding
    elif isinstance(padding, (tuple, list)) and len(padding) == 2:
        pp += ' 3=%d' % padding[1]
        pp += ' 13=%d' % padding[0]
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
    bottom_names = check_bottom_names(bottom_names)
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
        print('There is a sigmoid layer, maybe you can fuse it with its previous layer.')
        pp += 'Sigmoid\tlayer_%.8d\t1 1 %s %s\n' % (layer_id, bottom_names[0], top_names[0])
        layer_id += 1
        tensor_id += 1
    elif act_name == 'mish':
        print('There is a mish layer, maybe you can fuse it with its previous layer.')
        pp += 'Mish\tlayer_%.8d\t1 1 %s %s\n' % (layer_id, bottom_names[0], top_names[0])
        layer_id += 1
        tensor_id += 1
    elif act_name == 'hardsigmoid':
        pp += 'HardSigmoid\tlayer_%.8d\t1 1 %s %s 0=1.666667e-01 1=5.000000e-01\n' % (layer_id, bottom_names[0], top_names[0])
        layer_id += 1
        tensor_id += 1
    elif act_name == 'leaky_relu':
        print('There is a leaky_relu layer, maybe you can fuse it with its previous layer.')
        negative_slope = args['negative_slope']
        assert isinstance(negative_slope, float)
        pp += 'ReLU\tlayer_%.8d\t1 1 %s %s 0=%e\n' % (layer_id, bottom_names[0], top_names[0], negative_slope)
        layer_id += 1
        tensor_id += 1
    elif act_name == 'relu':
        print('There is a relu layer, maybe you can fuse it with its previous layer.')
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


def coordconcat(ncnn_data, bottom_names):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    top_names = create_top_names(ncnn_data, num=1)
    pp += 'CoordConcat\tlayer_%.8d\t1 1 %s %s 0=0\n' % (layer_id, bottom_names[0], top_names[0])
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def split(ncnn_data, bottom_names, num):
    bottom_names = check_bottom_names(bottom_names)
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
    bottom_names = check_bottom_names(bottom_names)
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
    bottom_names = check_bottom_names(bottom_names)
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
    fixbug0 = False
    if fixbug0:
        pp += ' 5=0'
    else:
        pp += ' 5=1'
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def really_reduction(ncnn_data, bottom_names, op, dims, keepdim=False):
    bottom_names = check_bottom_names(bottom_names)
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
    if reduce_all:
        pp += ' 1=1'
    else:
        pp += ' 1=0'
    # 被干掉的维的信息。
    pp += ' -23303=%d' % (len(dims),)
    for dim in dims:
        pp += ',%d' % (dim,)
    if keepdim:
        pp += ' 4=1'
    else:
        pp += ' 4=0'
    fixbug0 = False
    if fixbug0:
        pp += ' 5=0'
    else:
        pp += ' 5=1'
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def crop(ncnn_data, bottom_names, starts, ends, axes):
    bottom_names = check_bottom_names(bottom_names)
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
    bottom_names = check_bottom_names(bottom_names)
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
    bottom_names = check_bottom_names(bottom_names)
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
    bottom_names = check_bottom_names(bottom_names)
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
    elif perm == '(1, 0)':
        args = ' 0=1'
    elif perm == '(1, 0, 2, 3)':
        args = ' 0=11111'
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
    bottom_names = check_bottom_names(bottom_names)
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
    bottom_names = check_bottom_names(bottom_names)
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


def really_reshape(ncnn_data, bottom_names, shape):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    '''
    针对ncnn的CDHW格式的四维张量进行真正的reshape
    '''
    #
    assert isinstance(shape, (list, tuple))
    args = ''
    if len(shape) == 4:  # 此时shape代表ncnn中的CDHW
        args = ' 0=%d 1=%d 11=%d 2=%d' % (shape[3], shape[2], shape[1], shape[0])
    elif len(shape) == 3:  # 此时shape代表ncnn中的CHW
        args = ' 0=%d 1=%d 2=%d' % (shape[2], shape[1], shape[0])
    elif len(shape) == 2:  # 此时shape代表ncnn中的HW
        args = ' 0=%d 1=%d' % (shape[1], shape[0])
    elif len(shape) == 1:  # 此时shape代表ncnn中的W
        args = ' 0=%d' % (shape[0], )
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


def list_equal(arr1, arr2):
    if len(arr1) == 4:
        return arr1[0] == arr2[0] and arr1[1] == arr2[1] and arr1[2] == arr2[2] and arr1[3] == arr2[3]
    elif len(arr1) == 3:
        return arr1[0] == arr2[0] and arr1[1] == arr2[1] and arr1[2] == arr2[2]
    elif len(arr1) == 2:
        return arr1[0] == arr2[0] and arr1[1] == arr2[1]

def really_permute(ncnn_data, bottom_names, perm):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    assert isinstance(perm, (list, tuple))

    '''
    针对ncnn的CDHW格式的四维张量进行真正的permute
        // order_type
        // 0 = w h d c
        // 1 = h w d c
        // 2 = w d h c
        // 3 = d w h c
        // 4 = h d w c
        // 5 = d h w c
        // 6 = w h c d
        // 7 = h w c d
        // 8 = w c h d
        // 9 = c w h d
        //10 = h c w d
        //11 = c h w d
        //12 = w d c h
        //13 = d w c h
        //14 = w c d h
        //15 = c w d h
        //16 = d c w h
        //17 = c d w h
        //18 = h d c w
        //19 = d h c w
        //20 = h c d w
        //21 = c h d w
        //22 = d c h w
        //23 = c d h w
    但是维度排列你要倒着看。比如 CDHW格式 对应 上图的 w h d c，
    你要转成 DCHW格式(即python端perm==[1, 0, 2, 3])  对应 上图的 w h c d，所以参数填6。其它情况同理。
    
        // order_type
        // 0 = w h
        // 1 = h w
        
        // order_type
        // 0 = w h c
        // 1 = h w c
        // 2 = w c h
        // 3 = c w h
        // 4 = h c w
        // 5 = c h w
    '''
    #
    args = ''
    if len(perm) == 4:
        if list_equal(perm, [0, 1, 2, 3]):
            raise NotImplementedError("not implemented.")
        elif list_equal(perm, [1, 0, 2, 3]):
            args = ' 0=6'
        else:
            raise NotImplementedError("not implemented.")
    elif len(perm) == 3:
        if list_equal(perm, [0, 1, 2]):
            raise NotImplementedError("not implemented.")
        elif list_equal(perm, [2, 0, 1]):
            args = ' 0=4'
        else:
            raise NotImplementedError("not implemented.")
    elif len(perm) == 2:
        if list_equal(perm, [0, 1]):
            raise NotImplementedError("not implemented.")
        elif list_equal(perm, [1, 0]):
            args = ' 0=1'
        else:
            raise NotImplementedError("not implemented.")
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


def softmax(ncnn_data, bottom_names, dim):
    bottom_names = check_bottom_names(bottom_names)
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


def square(ncnn_data, bottom_names):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    top_names = create_top_names(ncnn_data, num=1)
    pp += 'Square\tlayer_%.8d\t1 1 %s %s' % (layer_id, bottom_names[0], top_names[0])
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def abs(ncnn_data, bottom_names):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    top_names = create_top_names(ncnn_data, num=1)
    pp += 'Abs\tlayer_%.8d\t1 1 %s %s' % (layer_id, bottom_names[0], top_names[0])
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def rsqrt(ncnn_data, bottom_names, eps=0.0, scale=None):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    top_names = create_top_names(ncnn_data, num=1)
    pp += 'Rsqrt\tlayer_%.8d\t1 1 %s %s 0=%e' % (layer_id, bottom_names[0], top_names[0], eps)
    if scale is not None:
        pp += ' 1=%e' % (scale, )
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def sqrt(ncnn_data, bottom_names, eps=0.0):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    top_names = create_top_names(ncnn_data, num=1)
    pp += 'Sqrt\tlayer_%.8d\t1 1 %s %s 0=%e' % (layer_id, bottom_names[0], top_names[0], eps)
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def sin(ncnn_data, bottom_names, scale=1.0):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    top_names = create_top_names(ncnn_data, num=1)
    pp += 'Sin\tlayer_%.8d\t1 1 %s %s 0=%e' % (layer_id, bottom_names[0], top_names[0], scale)
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def clamp(ncnn_data, bottom_names, min_v=0.0, max_v=1.0):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    top_names = create_top_names(ncnn_data, num=1)
    pp += 'Clamp\tlayer_%.8d\t1 1 %s %s 0=%e 1=%e' % (layer_id, bottom_names[0], top_names[0], min_v, max_v)
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def lerp(ncnn_data, bottom_names):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    top_names = create_top_names(ncnn_data, num=1)
    num = len(bottom_names)
    pp += 'Lerp\tlayer_%.8d\t%d 1' % (layer_id, num)
    for i in range(num):
        pp += ' %s' % bottom_names[i]
    pp += ' %s' % top_names[0]
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def StyleMixingSwitcher(ncnn_data, bottom_names, ws_i=0):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    top_names = create_top_names(ncnn_data, num=1)
    num = len(bottom_names)
    pp += 'StyleMixingSwitcher\tlayer_%.8d\t%d 1' % (layer_id, num)
    for i in range(num):
        pp += ' %s' % bottom_names[i]
    pp += ' %s' % top_names[0]
    pp += ' 0=%d' % (ws_i,)
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def MulConstant(ncnn_data, bottom_names, scale=1.0, bias=0.0):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    top_names = create_top_names(ncnn_data, num=1)
    pp += 'MulConstant\tlayer_%.8d\t1 1 %s %s 0=%e 1=%e' % (layer_id, bottom_names[0], top_names[0], scale, bias)
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def shell(ncnn_data, bottom_names, weight, bias, ncnn_weight_dims=4):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    '''
    weight 是2维张量时，要求传入的weight reshape成 CD11形状(ncnn中的形状也是CD11)，bias形状要求是[C, ](ncnn中的形状是111C)；
    weight 是4维张量时，要求传入的weight是CDHW形状(ncnn中的形状也是CDHW)，bias形状要求是[C, ](ncnn中的形状是111C)；
    '''

    w_dims = len(weight.shape)
    assert w_dims == 4
    C, D, H, W = weight.shape
    num = 1
    if bias is not None:
        num = 2


    top_names = create_top_names(ncnn_data, num=num)
    pp += 'Shell\tlayer_%.8d\t1 %d %s' % (layer_id, num, bottom_names[0])
    for i in range(num):
        pp += ' %s' % top_names[i]

    pp += ' 2=%d' % C
    pp += ' 11=%d' % D
    pp += ' 1=%d' % H
    pp += ' 0=%d' % W
    pp += ' 3=%d' % ncnn_weight_dims
    if bias is not None:
        pp += ' 5=1'
    else:
        pp += ' 5=0'
    w_ele_num = C * D * H * W
    pp += ' 6=%d' % w_ele_num
    pp += '\n'
    layer_id += 1
    tensor_id += num

    # 卷积层写入权重。参考了onnx2ncnn，开头写个0
    s = struct.pack('i', 0)
    bp.write(s)

    conv_w = weight.cpu().detach().numpy()

    if w_dims == 4:
        for i1 in range(C):
            for i2 in range(D):
                for i3 in range(H):
                    for i4 in range(W):
                        s = struct.pack('f', conv_w[i1][i2][i3][i4])
                        bp.write(s)
    if bias is not None:
        conv_b = bias.cpu().detach().numpy()
        for i1 in range(C):
            s = struct.pack('f', conv_b[i1])
            bp.write(s)
    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def Fmatmul(ncnn_data, bottom_names, weight_shape):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    '''
    注意！Fmatmul层借鉴了卷积层的代码，当kH==1和kW==1时，调用InnerProduct层。
    卷积层的权重的形状是[out_C, in_C, kH, kW]
    所以必须要保证传入Fmatmul层的w的形状是[out_C, in_C]而不能是[in_C, out_C]，否则会计算出错误结果！
    '''

    num_input = len(bottom_names)
    assert num_input in [2, 3]
    bias_term = 1
    if num_input == 2:
        bias_term = 0

    w_dims = len(weight_shape)
    assert w_dims in [2, ]
    if w_dims == 2:
        out_C, in_C = weight_shape

    top_names = create_top_names(ncnn_data, num=1)
    pp += 'Fmatmul\tlayer_%.8d\t%d 1' % (layer_id, num_input)
    for i in range(num_input):
        pp += ' %s' % bottom_names[i]
    pp += ' %s' % top_names[0]
    pp += ' 0=%d' % out_C
    pp += ' 1=%d' % bias_term
    w_ele_num = out_C * in_C
    pp += ' 2=%d' % w_ele_num
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def BiasAct(ncnn_data, bottom_names, act_type, alpha, gain, clamp):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    num_input = len(bottom_names)
    top_names = create_top_names(ncnn_data, num=1)
    pp += 'BiasAct\tlayer_%.8d\t%d 1' % (layer_id, num_input)
    for i in range(num_input):
        pp += ' %s' % bottom_names[i]
    pp += ' %s' % top_names[0]
    pp += ' 0=%d' % act_type
    pp += ' 1=%e' % alpha
    pp += ' 2=%e' % gain
    pp += ' 3=%e' % clamp
    if num_input == 1:
        pp += ' 4=0'
    elif num_input == 2:
        pp += ' 4=1'
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def F4DOp1D(ncnn_data, bottom_names, dim, op):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    op_id = -1
    if op == 'Mul':
        op_id = 0
    elif op == 'Div':
        op_id = 1
    elif op == 'Add':
        op_id = 2
    elif op == 'Sub':
        op_id = 3
    else:
        raise NotImplementedError("not implemented.")

    num_input = len(bottom_names)
    top_names = create_top_names(ncnn_data, num=1)
    pp += 'F4DOp1D\tlayer_%.8d\t%d 1' % (layer_id, num_input)
    for i in range(num_input):
        pp += ' %s' % bottom_names[i]
    pp += ' %s' % top_names[0]
    pp += ' 0=%d' % dim
    pp += ' 1=%d' % op_id
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def AddNoise(ncnn_data, bottom_names):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    num_input = len(bottom_names)
    top_names = create_top_names(ncnn_data, num=1)
    pp += 'AddNoise\tlayer_%.8d\t%d 1' % (layer_id, num_input)
    for i in range(num_input):
        pp += ' %s' % bottom_names[i]
    pp += ' %s' % top_names[0]
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def Fconv2d(ncnn_data, bottom_names, stride=1, padding=0, dilation=1, groups=1):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    top_names = create_top_names(ncnn_data, num=1)
    num = len(bottom_names)
    pp += 'Convolution\tlayer_%.8d\t%d 1' % (layer_id, num)
    for i in range(num):
        pp += ' %s' % bottom_names[i]
    pp += ' %s' % top_names[0]

    pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
    if isinstance(padding, int):
        pad_left = padding
        pad_right = padding
        pad_top = padding
        pad_bottom = padding
    elif isinstance(padding, list) or isinstance(padding, tuple):
        if len(padding) == 2:
            pad_left = padding[1]
            pad_right = padding[1]
            pad_top = padding[0]
            pad_bottom = padding[0]
        elif len(padding) == 4:
            pad_left = padding[2]
            pad_right = padding[3]
            pad_top = padding[0]
            pad_bottom = padding[1]
        else:
            raise NotImplementedError("not implemented.")
    else:
        raise NotImplementedError("not implemented.")
    pp += ' 4=%d' % pad_left
    pp += ' 15=%d' % pad_right
    pp += ' 14=%d' % pad_top
    pp += ' 16=%d' % pad_bottom

    stride_h, stride_w = 1, 1
    if isinstance(stride, int):
        stride_h = stride
        stride_w = stride
    elif isinstance(stride, list) or isinstance(stride, tuple):
        if len(padding) == 2:
            stride_h = stride[0]
            stride_w = stride[1]
        else:
            raise NotImplementedError("not implemented.")
    else:
        raise NotImplementedError("not implemented.")
    pp += ' 3=%d' % stride_w
    pp += ' 13=%d' % stride_h

    if num == 3:
        pp += ' 5=1'
    else:
        pp += ' 5=0'

    # dynamic_weight
    pp += ' 19=1'
    assert dilation == 1
    assert groups == 1
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def Fconv2d_depthwise(ncnn_data, bottom_names, stride=1, padding=0, dilation=1, groups=1):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    top_names = create_top_names(ncnn_data, num=1)
    num = len(bottom_names)
    pp += 'ConvolutionDepthWise\tlayer_%.8d\t%d 1' % (layer_id, num)
    for i in range(num):
        pp += ' %s' % bottom_names[i]
    pp += ' %s' % top_names[0]

    pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
    if isinstance(padding, int):
        pad_left = padding
        pad_right = padding
        pad_top = padding
        pad_bottom = padding
    elif isinstance(padding, list) or isinstance(padding, tuple):
        if len(padding) == 2:
            pad_left = padding[1]
            pad_right = padding[1]
            pad_top = padding[0]
            pad_bottom = padding[0]
        elif len(padding) == 4:
            pad_left = padding[2]
            pad_right = padding[3]
            pad_top = padding[0]
            pad_bottom = padding[1]
        else:
            raise NotImplementedError("not implemented.")
    else:
        raise NotImplementedError("not implemented.")
    pp += ' 4=%d' % pad_left
    pp += ' 15=%d' % pad_right
    pp += ' 14=%d' % pad_top
    pp += ' 16=%d' % pad_bottom

    stride_h, stride_w = 1, 1
    if isinstance(stride, int):
        stride_h = stride
        stride_w = stride
    elif isinstance(stride, list) or isinstance(stride, tuple):
        if len(padding) == 2:
            stride_h = stride[0]
            stride_w = stride[1]
        else:
            raise NotImplementedError("not implemented.")
    else:
        raise NotImplementedError("not implemented.")
    pp += ' 3=%d' % stride_w
    pp += ' 13=%d' % stride_h

    if num == 3:
        pp += ' 5=1'
    else:
        pp += ' 5=0'

    # dynamic_weight
    pp += ' 19=1'
    assert dilation == 1
    pp += ' 7=%d' % groups
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def Fconv_transpose2d(ncnn_data, bottom_names, weight_shape, stride=1, padding=0, output_padding=0, dilation=1, groups=1):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    top_names = create_top_names(ncnn_data, num=1)
    num = len(bottom_names)
    pp += 'FconvTranspose2d\tlayer_%.8d\t%d 1' % (layer_id, num)
    for i in range(num):
        pp += ' %s' % bottom_names[i]
    pp += ' %s' % top_names[0]

    pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
    if isinstance(padding, int):
        pad_left = padding
        pad_right = padding
        pad_top = padding
        pad_bottom = padding
    elif isinstance(padding, list) or isinstance(padding, tuple):
        if len(padding) == 2:
            pad_left = padding[1]
            pad_right = padding[1]
            pad_top = padding[0]
            pad_bottom = padding[0]
        elif len(padding) == 4:
            pad_left = padding[2]
            pad_right = padding[3]
            pad_top = padding[0]
            pad_bottom = padding[1]
        else:
            raise NotImplementedError("not implemented.")
    else:
        raise NotImplementedError("not implemented.")
    pp += ' 4=%d' % pad_left
    pp += ' 15=%d' % pad_right
    pp += ' 14=%d' % pad_top
    pp += ' 16=%d' % pad_bottom

    stride_h, stride_w = 1, 1
    if isinstance(stride, int):
        stride_h = stride
        stride_w = stride
    elif isinstance(stride, list) or isinstance(stride, tuple):
        if len(padding) == 2:
            stride_h = stride[0]
            stride_w = stride[1]
        else:
            raise NotImplementedError("not implemented.")
    else:
        raise NotImplementedError("not implemented.")
    pp += ' 3=%d' % stride_w
    pp += ' 13=%d' % stride_h

    if num == 3:
        pp += ' 5=1'
    else:
        pp += ' 5=0'
    out_C, in_C, kH, kW = weight_shape
    w_ele_num = out_C * in_C * kH * kW
    pp += ' 6=%d' % w_ele_num
    pp += ' 11=%d' % kH
    pp += ' 1=%d' % kW
    pp += ' 0=%d' % out_C
    pp += ' 31=%d' % in_C

    assert dilation == 1
    assert groups == 1
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def pad(ncnn_data, bottom_names, top=0, bottom=0, left=0, right=0, front=0, behind=0, value=0.0, mode="constant"):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    assert mode in ["constant", "edge", "reflect"]

    top_names = create_top_names(ncnn_data, num=1)
    pp += 'Padding\tlayer_%.8d\t1 1 %s %s' % (layer_id, bottom_names[0], top_names[0])
    if mode == "constant":
        pp += ' 4=0'
    elif mode == "edge":
        pp += ' 4=1'
    elif mode == "reflect":
        pp += ' 4=2'

    pp += ' 5=%e' % value
    pp += ' 0=%d' % top
    pp += ' 1=%d' % bottom
    pp += ' 2=%d' % left
    pp += ' 3=%d' % right
    pp += ' 7=%d' % front
    pp += ' 8=%d' % behind
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def down2(ncnn_data, bottom_names):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    top_names = create_top_names(ncnn_data, num=1)
    pp += 'Down2\tlayer_%.8d\t1 1 %s %s' % (layer_id, bottom_names[0], top_names[0])
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def up2(ncnn_data, bottom_names):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    top_names = create_top_names(ncnn_data, num=1)
    pp += 'Up2\tlayer_%.8d\t1 1 %s %s' % (layer_id, bottom_names[0], top_names[0])
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def up4(ncnn_data, bottom_names):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    top_names = create_top_names(ncnn_data, num=1)
    pp += 'Up4\tlayer_%.8d\t1 1 %s %s' % (layer_id, bottom_names[0], top_names[0])
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def Transforms(ncnn_data, bottom_names):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    top_names = create_top_names(ncnn_data, num=1)
    pp += 'Transforms\tlayer_%.8d\t1 1 %s %s' % (layer_id, bottom_names[0], top_names[0])
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


def StyleganPost(ncnn_data, bottom_names):
    bottom_names = check_bottom_names(bottom_names)
    bp = ncnn_data['bp']
    pp = ncnn_data['pp']
    layer_id = ncnn_data['layer_id']
    tensor_id = ncnn_data['tensor_id']

    top_names = create_top_names(ncnn_data, num=1)
    pp += 'StyleganPost\tlayer_%.8d\t1 1 %s %s' % (layer_id, bottom_names[0], top_names[0])
    pp += '\n'
    layer_id += 1
    tensor_id += 1

    ncnn_data['bp'] = bp
    ncnn_data['pp'] = pp
    ncnn_data['layer_id'] = layer_id
    ncnn_data['tensor_id'] = tensor_id
    return top_names


