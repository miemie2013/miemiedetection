import paddle
import numpy as np
import paddle.nn.functional as F

'''
测试C++版的卷积层
'''

def write_line(name, ndarray, dims, content):
    content += '%s '%name
    print('uuuuuuuuuuuuuuuuuuuuuuuuuuuuuu')
    if dims==4:
        for i in range(ndarray.shape[0]):
            for j in range(ndarray.shape[1]):
                for k in range(ndarray.shape[2]):
                    for l in range(ndarray.shape[3]):
                        content += '%f,' % ndarray[i, j, k, l]
    elif dims==1:
        for i in range(ndarray.shape[0]):
            content += '%f,' % ndarray[i]
    content = content[:-1]+'\n'
    print('uuuuuuuuuuuuuuuuuuuuuuuuuuuuuu')
    return content


# x = paddle.uniform([8, 3, 256, 256], min=-1.0, max=1.0, dtype='float32')
x = paddle.uniform([2, 1, 1, 1], min=-1.0, max=1.0, dtype='float32')

dic = {}
content = ''
dic['x'] = x.numpy()
print('uuuuuuuuuuuuuuuuuuuuuuuuuuuuuu')
content = write_line('x', dic['x'], 4, content)

# 测试用例1
input_dim = 1
filters = 1
filter_size = 1
stride = 1
padding = 0
groups = 1
bias_attr = False

# 测试用例2
# input_dim = 3
# filters = 2
# filter_size = 3
# stride = 1
# padding = 1
# groups = 1
# bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Normal())


conv = paddle.nn.Conv2D(
                in_channels=input_dim,
                out_channels=filters,
                kernel_size=filter_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias_attr=bias_attr)
conv.eval()


dic['w'] = conv.weight.numpy()
content = write_line('w', dic['w'], 4, content)
if bias_attr == False:
    pass
else:
    dic['b'] = conv.bias.numpy()
    content = write_line('b', dic['b'], 1, content)

out = conv(x)

dic['out'] = out.numpy()
print('uuuuuuuuuuuuuuuuuuuuuuuuuuuuuu2222222222222222')
content = write_line('out', dic['out'], 4, content)
np.savez('conv2d', **dic)

with open('conv2d.txt', 'w', encoding='utf-8') as f:
    f.write(content)
    f.close()


print()









