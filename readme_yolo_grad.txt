
----------------------- 梯度对齐 -----------------------
1.数据集为voc2012_train.json、voc2012_val.json

2.批大小为8，每张图片学习率为0.0004 / 8.0，Warmup steps=714, start_factor = 0.5, freeze_at: 3 （resnet全部权重不可以训练）

3.DropBlock改成直接返回，避免随机数的影响：
class DropBlock(nn.Layer):
    ...
    def forward(self, x):
        return x

4.trainer.py下面代码解除注释
            # 对齐梯度用
            # if (self.iter + 1) == 10:
            #     if self.rank == 0:
            #         self.save_ckpt(ckpt_name="%d" % (self.epoch + 1))
            #     break
对下面所有的以if self.align_grad:开头的代码块解除注释
if self.align_grad:
    xxx

python tools/convert_weights.py -f exps/ppyolo/ppyolo_r50vd_aligngrad.py -c 00.pdema -oc 00.pth -nc 20

python tools/convert_weights.py -f exps/ppyolo/ppyolo_r50vd_aligngrad.py -c 09.pdema -oc 09.pth -nc 20

python tools/convert_weights.py -f exps/ppyolo/ppyolo_r50vd_aligngrad.py -c 09.pdparams -oc 09_ema.pth -nc 20




1机1卡：
CUDA_VISIBLE_DEVICES=0
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_aligngrad.py -d 1 -b 8 -eb 1 -c 00.pth


CUDA_VISIBLE_DEVICES=0,1
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_aligngrad.py -d 2 -b 8 -eb 2 -c 00.pth


CUDA_VISIBLE_DEVICES=0
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_aligngrad.py --dist-url tcp://192.168.0.107:12319 --num_machines 2 --machine_rank 1 -b 8 -eb 2 -c 00.pth



python diff_weights.py --cp1 09.pth --cp2 PPYOLO_outputs/ppyolo_r50vd_aligngrad/1.pth --d_value 0.0005


python diff_weights.py --cp1 09.pth --cp2 00.pth --d_value 0.0005


----------------------- 自己和自己对齐 -----------------------
1.数据集为voc2012_train.json、voc2012_val.json

2.批大小为8，每张图片学习率为0.0004 / 8.0，Warmup steps=714, start_factor = 0.5, freeze_at: 3 （resnet全部权重不可以训练）

3.DropBlock改成直接返回，避免随机数的影响：
class DropBlock(nn.Layer):
    ...
    def forward(self, x):
        return x

4.trainer.py下面代码注释掉
            # 对齐梯度用
            # if (self.iter + 1) == 10:
            #     if self.rank == 0:
            #         self.save_ckpt(ckpt_name="%d" % (self.epoch + 1))
            #     break
注释掉下面所有的以if self.align_grad:开头的代码块
if self.align_grad:
    xxx

对下面所有的以if self.save_npz:开头的代码块解除注释（包括它带的else）
if self.save_npz:
    xxx


1机1卡：
CUDA_VISIBLE_DEVICES=0
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_aligngrad.py -d 1 -b 8 -eb 1 -c ppyolo_r50vd_2x.pth --save_npz 1

python tools/train.py -f exps/ppyolo/ppyolo_r50vd_aligngrad.py -d 1 -b 8 -eb 1 -c PPYOLO_outputs/ppyolo_r50vd_aligngrad/000.pth --save_npz 0


python diff_weights.py --cp1 PPYOLO_outputs/ppyolo_r50vd_aligngrad/1__.pth --cp2 PPYOLO_outputs/ppyolo_r50vd_aligngrad/1.pth --d_value 0.0005



CUDA_VISIBLE_DEVICES=0,1
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_aligngrad.py -d 2 -b 8 -eb 2 -c ppyolo_r50vd_2x.pth --save_npz 1

python tools/train.py -f exps/ppyolo/ppyolo_r50vd_aligngrad.py -d 2 -b 8 -eb 2 -c PPYOLO_outputs/ppyolo_r50vd_aligngrad/000.pth --save_npz 0


python diff_weights.py --cp1 PPYOLO_outputs/ppyolo_r50vd_aligngrad/1__.pth --cp2 PPYOLO_outputs/ppyolo_r50vd_aligngrad/1.pth --d_value 0.0005



----------------------- 进阶：单卡和多卡进行对齐（用 单卡批大小8 实现 双卡总批大小8 的效果） -----------------------
先保存2卡总批大小8（每张卡上批大小4）的结果，ppyolo_r50vd_aligngrad.py里的
多尺度训练的sizes改成[416]，因为每张卡上的图片分辨率可能是不一样的！！！在和单卡对齐时会无法拼接成一个图片张量，
所以只使用416分辨率。再跑：
CUDA_VISIBLE_DEVICES=0,1
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_aligngrad.py -d 2 -b 8 -eb 2 -c ppyolo_r50vd_2x.pth --save_npz 1


或者2机2卡：
CUDA_VISIBLE_DEVICES=0
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_aligngrad.py --dist-url tcp://192.168.0.107:12312 --num_machines 2 --machine_rank 0 -b 8 -eb 2 -c ppyolo_r50vd_2x.pth --save_npz 1



跑完之后，ppyolo_r50vd_aligngrad.py里 多尺度训练的
sizes改回[320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
把1__.pth改名成1___2gpu.pth
再修改下面这些代码：

1.设置 trainer.py 的
对下面所有的以if self.align_2gpu_1gpu:开头的代码块解除注释
if self.align_2gpu_1gpu:
    xxx

输入以下命令验证是否已经对齐：
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_aligngrad.py -d 1 -b 8 -eb 1 -c PPYOLO_outputs/ppyolo_r50vd_aligngrad/000.pth --save_npz 0


python diff_weights.py --cp1 PPYOLO_outputs/ppyolo_r50vd_aligngrad/1___2gpu.pth --cp2 PPYOLO_outputs/ppyolo_r50vd_aligngrad/1.pth --d_value 0.0005



会发现差距很小。而且，我全程没有修改学习率参数，不管单卡还是双卡，总批大小都是8，不需要改学习率。
也验证了同步bn的正确性。


而且发现一个细节，计算损失时，是每张图片的内部样本损失求和，再求每张图片的损失的平均值，比如loss_xy是这样计算的：
loss_xy = loss_xy.sum([1, 2, 3, 4]).mean()
因此，
当使用 单卡总批大小8 训练时，.mean()相当 / 8.0，伪代码即 loss_xy /= 8.0
当使用 双卡总批大小8 训练时，.mean()相当 / 4.0，伪代码即 loss_xy_gpu0_ = loss_xy_gpu0 / 4.0, loss_xy_gpu1_ = loss_xy_gpu1 / 4.0,
loss_xy = (loss_xy_gpu0_ + loss_xy_gpu1_) / 2.0 = (loss_xy_gpu0 + loss_xy_gpu1) / 8.0
单卡和双卡的损失值是一样的。

在ppgan的test_DDP/下验证过，多卡训练时,每一张卡上的梯度（损失）是求平均值而并不是求和。test_DDP/需要修改学习率，是因为它的损失是
loss = dstyles2_dws.sum() + styles2.sum()
是每个ws的损失求和，如果改成像PPYOLO这样
loss = dstyles2_dws.sum([1]).mean() + styles2.sum([1]).mean()
就不用修改学习率。

所以if self.align_2gpu_1gpu:代码段内部融合卡0的.npz和卡1的.npz时，
                                if 'loss_' in key:
                                    v3 = (v1 + v2) / 2.


