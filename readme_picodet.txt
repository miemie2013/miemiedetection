

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


----------------------- paddle语法转pytorch -----------------------

class SPP(nn.Layer):
    def __init__(...):
        super(SPP, self).__init__()
        self.pool = []
        self.data_format = data_format
        for i, size in enumerate(pool_size):
            pool = self.add_sublayer(
                'pool{}'.format(i),
                nn.MaxPool2D(
                    kernel_size=size,
                    stride=1,
                    padding=size // 2,
                    data_format=data_format,
                    ceil_mode=False))
            self.pool.append(pool)
        self.conv = ConvBNLayer(ch_in, ch_out, k, padding=k // 2, act=act)

转成（见 custom_pan.py）:

class SPP(nn.Module):
    def __init__(...):
        super(SPP, self).__init__()
        self.pool = []
        self.data_format = data_format
        for i, size in enumerate(pool_size):
            name = 'pool{}'.format(i)
            pool = nn.MaxPool2d(
                    kernel_size=size,
                    stride=1,
                    padding=size // 2,
                    ceil_mode=False)
            self.add_module(name, pool)
            self.pool.append(pool)
        self.conv = ConvBNLayer(ch_in, ch_out, k, padding=k // 2, act=act, act_name=act_name)

即
            pool = self.add_sublayer('pool{}'.format(i), nn.MaxPool2D(...))
拆分成
            name = 'pool{}'.format(i)
            pool = nn.MaxPool2d(...)
            self.add_module(name, pool)
这3句代码




----------------------- 转换权重 -----------------------
wget https://paddledet.bj.bcebos.com/models/picodet_s_416_coco_lcnet.pdparams
wget https://paddledet.bj.bcebos.com/models/picodet_m_416_coco_lcnet.pdparams


python tools/convert_weights.py -f exps/picodet/picodet_s_416_coco_lcnet.py -c picodet_s_416_coco_lcnet.pdparams -oc picodet_s_416_coco_lcnet.pth -nc 80

python tools/convert_weights.py -f exps/picodet/picodet_m_416_coco_lcnet.py -c picodet_m_416_coco_lcnet.pdparams -oc picodet_m_416_coco_lcnet.pth -nc 80


python tools/convert_weights.py -f exps/ppyoloe/ppyoloe_crn_s_300e_coco.py -c CSPResNetb_s_pretrained.pdparams -oc CSPResNetb_s_pretrained.pth -nc 80 --only_backbone True



----------------------- 预测 -----------------------
python tools/demo.py image -f exps/picodet/picodet_s_416_coco_lcnet.py -c picodet_s_416_coco_lcnet.pth --path assets/000000000019.jpg --conf 0.15 --tsize 416 --save_result --device gpu







----------------------- 训练 -----------------------
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 8 -eb 4





----------------------- 迁移学习，带上-c（--ckpt）参数读取预训练模型。 -----------------------
后台启动：
nohup xxx     > ppyolo.log 2>&1 &


- - - - - - - - - - - - - - - - - - - - - -
迁移学习（不冻结骨干网络）:（可以加--fp16， 但是picodet没有用自动混合精度训练。-eb表示验证时的批大小）
python tools/train.py -f exps/picodet/picodet_s_416_voc2012.py -d 1 -b 8 -eb 2 -c picodet_s_416_coco_lcnet.pth

python tools/eval.py -f exps/picodet/picodet_s_416_voc2012.py -d 1 -b 4 -c PPYOLOE_outputs/ppyoloe_crn_s_voc2012/16.pth --conf 0.01 --tsize 640

python tools/demo.py image -f exps/picodet/picodet_s_416_voc2012.py -c PPYOLOE_outputs/ppyoloe_crn_s_voc2012/16.pth --path assets/000000000019.jpg --conf 0.15 --tsize 640 --save_result --device gpu


1机2卡训练：(发现一个隐藏知识点：获得损失（训练）、推理 都要放在模型的forward()中进行，否则DDP会计算错误结果。)
export CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/picodet/picodet_s_416_voc2012.py -d 2 -b 8 -eb 2 -c picodet_s_416_coco_lcnet.pth     > picodet_s_416.log 2>&1 &

tail -n 20 picodet_s_416.log



实测 picodet_s_416 的AP(0.50:0.95)可以到达0.48+、AP(0.50)可以到达0.68+、AP(small)可以到达0.15+。





----------------------- 恢复训练（加上参数--resume） -----------------------
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 8 -eb 4 -c PPYOLO_outputs/ppyolo_r50vd_2x/13.pth --resume


python tools/train.py -f exps/ppyolo/ppyolo_r18vd.py -d 1 -b 16 -eb 8 -c PPYOLO_outputs/ppyolo_r18vd/7.pth --resume





----------------------- 评估 -----------------------
python tools/eval.py -f exps/picodet/picodet_s_416_coco_lcnet.py -d 1 -b 4 -c picodet_s_416_coco_lcnet.pth --conf 0.025 --tsize 416

Average forward time: 8.53 ms, Average NMS time: 0.00 ms, Average inference time: 8.54 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.320
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.469
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.339
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.111
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.341
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.517
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.277
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.432
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.462
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.185
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.519
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.706





----------------------- 导出为ONNX -----------------------




----------------------- 导出为TensorRT -----------------------


----------------------- 复现COCO上的精度 -----------------------

(5)ppyoloe_crn_s_300e_coco
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nohup python tools/train.py -f exps/ppyoloe/ppyoloe_crn_s_300e_coco.py -d 8 -b 256 -eb 16 -c CSPResNetb_s_pretrained.pth     > ppyoloe_crn_s_300e_coco_8gpu.log 2>&1 &

只有双卡的时候：
export CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/ppyoloe/ppyoloe_crn_s_300e_coco.py -d 2 -b 64 -eb 16 -c CSPResNetb_s_pretrained.pth     > ppyoloe_crn_s_300e_coco_2gpu.log 2>&1 &




