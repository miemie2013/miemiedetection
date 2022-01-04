import paddle
import numpy as np
import paddle.nn.functional as F

'''
ClipGradByGlobalNorm，PPYOLOv2 中使用。先跑这个脚本，再跑test2_ClipGradByGlobalNorm_pytorch.py。
算法很简单，如果梯度的模大于clip_norm，那么先将梯度化为单位向量（方向不变），再乘以clip_norm。
'''



x = paddle.uniform([8, 10], min=-1.0, max=1.0, dtype='float32')

dic = {}
dic['x'] = x.numpy()


# 只裁剪weight，不裁剪bias
linear = paddle.nn.Linear(in_features=10, out_features=2,
                          weight_attr=paddle.ParamAttr(need_clip=True),
                          bias_attr=paddle.ParamAttr(need_clip=False))

dic['w'] = linear.weight.numpy()
dic['b'] = linear.bias.numpy()

out = linear(x)
loss = 1000.0 * paddle.mean(out)  # 乘以1000.0放大损失，使得梯度的模很大
dic['loss'] = loss.numpy()
loss.backward()

clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), grad_clip=clip)
sdg.step()

dic['w_2'] = linear.weight.numpy()
dic['b_2'] = linear.bias.numpy()
np.savez('ClipGradByGlobalNorm', **dic)
print()









