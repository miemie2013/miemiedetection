

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
wget https://bj.bcebos.com/v1/paddledet/models/rtdetr_r18vd_dec3_6x_coco.pdparams
wget https://bj.bcebos.com/v1/paddledet/models/rtdetr_r34vd_dec4_6x_coco.pdparams
wget https://bj.bcebos.com/v1/paddledet/models/rtdetr_r50vd_m_6x_coco.pdparams
wget https://bj.bcebos.com/v1/paddledet/models/rtdetr_r50vd_6x_coco.pdparams
wget https://bj.bcebos.com/v1/paddledet/models/rtdetr_r101vd_6x_coco.pdparams
wget https://bj.bcebos.com/v1/paddledet/models/rtdetr_hgnetv2_l_6x_coco.pdparams
wget https://bj.bcebos.com/v1/paddledet/models/rtdetr_hgnetv2_x_6x_coco.pdparams

wget https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_vd_ssld_v2_pretrained.pdparams

复现训练时请仔细核对每个参数的 lr、 L2Decay
forward流程：
骨干网络出来3个张量，形状是[N, 512, 80, 80], [N, 1024, 40, 40], [N, 2048, 20, 20],
进入 HybridEncoder,
先分别用3个conv(1x1卷积)+bn进行降维(相关层名字是input_proj)，形状变成[N, 256, 80, 80], [N, 256, 40, 40], [N, 256, 20, 20], 256是hidden_dim,


encoder_layer(TransformerLayer) 被放进 HybridEncoder 的 self.encoder(nn.ModuleList)
self.encoder里有1个元素，类型是 TransformerEncoder(encoder_layer, num_encoder_layers)


python tools/convert_weights.py -f exps/rtdetr/rtdetr_r50vd_6x_coco.py -c rtdetr_r50vd_6x_coco.pdparams -oc rtdetr_r50vd_6x_coco.pth -nc 80


python tools/convert_weights.py -f exps/rtdetr/rtdetr_r50vd_6x_coco.py -c ResNet50_vd_ssld_v2_pretrained.pdparams -oc ResNet50_vd_ssld_v2_pretrained.pth -nc 80 --only_backbone True




----------------------- 预测 -----------------------
python tools/demo.py image -f exps/picodet/picodet_s_416_coco_lcnet.py -c picodet_s_416_coco_lcnet.pth --path assets/000000000019.jpg --conf 0.15 --tsize 416 --save_result --device gpu

python tools/demo.py image -f exps/picodet/picodet_m_416_coco_lcnet.py -c picodet_m_416_coco_lcnet.pth --path assets/000000000019.jpg --conf 0.15 --tsize 416 --save_result --device gpu

python tools/demo.py image -f exps/picodet/picodet_l_320_coco_lcnet.py -c picodet_l_320_coco_lcnet.pth --path assets/000000000019.jpg --conf 0.15 --tsize 320 --save_result --device gpu
python tools/demo.py image -f exps/picodet/picodet_l_416_coco_lcnet.py -c picodet_l_416_coco_lcnet.pth --path assets/000000000019.jpg --conf 0.15 --tsize 416 --save_result --device gpu
python tools/demo.py image -f exps/picodet/picodet_l_640_coco_lcnet.py -c picodet_l_640_coco_lcnet.pth --path assets/000000000019.jpg --conf 0.15 --tsize 640 --save_result --device gpu







----------------------- 训练 -----------------------




----------------------- 迁移学习，带上-c（--ckpt）参数读取预训练模型。 -----------------------
后台启动：
nohup xxx     > ppyolo.log 2>&1 &


- - - - - - - - - - - - - - - - - - - - - -
迁移学习（不冻结骨干网络）:（可以加--fp16， 但是picodet没有用自动混合精度训练。-eb表示验证时的批大小）
python tools/train.py -f exps/picodet/picodet_s_416_voc2012.py -d 1 -b 48 -eb 24 -w 4 -ew 4 -c picodet_s_416_coco_lcnet.pth

python tools/eval.py -f exps/picodet/picodet_s_416_voc2012.py -d 1 -b 8 -c 16.pth --conf 0.025 --tsize 416

python tools/demo.py image -f exps/picodet/picodet_s_416_voc2012.py -c 16.pth --path assets/000000000019.jpg --conf 0.15 --tsize 416 --save_result --device gpu


1机2卡训练：(发现一个隐藏知识点：获得损失（训练）、推理 都要放在模型的forward()中进行，否则DDP会计算错误结果。)
export CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/picodet/picodet_s_416_voc2012.py -d 2 -b 48 -eb 24 -w 4 -ew 4 -c picodet_s_416_coco_lcnet.pth     > picodet_s_416.log 2>&1 &

tail -n 20 picodet_s_416.log



实测 picodet_s_416 的AP(0.50:0.95)可以到达0.487+、AP(0.50)可以到达0.676+、AP(small)可以到达0.064+。


如果修改 exps/picodet/picodet_s_416_voc2012.py 的 self.head['static_assigner_epoch'] = 499，即全程使用 ATSS 分配正负样本，则
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.464
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.656
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.497
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.070
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.238
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.583
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.415
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.591
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.616
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.177
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.438
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.720


如果修改 exps/picodet/picodet_s_416_voc2012.py 的 self.head['static_assigner_epoch'] = -4，即全程使用 TaskAlignedAssigner 分配正负样本，则
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.493
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.685
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.525
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.067
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.253
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.616
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.434
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.622
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.652
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.197
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.473
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.758



python tools/train.py -f exps/picodet/picodet_s_416_posass_voc2012.py -d 1 -b 48 -eb 24 -w 4 -ew 4 -c picodet_s_416_coco_lcnet.pth

export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1
python tools/train.py -f exps/picodet/picodet_s_416_posass_voc2012.py -d 2 -b 48 -eb 24 -w 4 -ew 4 -c picodet_s_416_coco_lcnet.pth


- - - - - - - - - - - - - - - - - - - - - -
迁移学习（不冻结骨干网络）:（可以加--fp16， 但是picodet没有用自动混合精度训练。-eb表示验证时的批大小）
1机2卡训练：(发现一个隐藏知识点：获得损失（训练）、推理 都要放在模型的forward()中进行，否则DDP会计算错误结果。)
export CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/picodet/picodet_m_640_voc2012.py -d 2 -b 24 -eb 8 -w 4 -ew 4 -c picodet_m_416_coco_lcnet.pth     > picodet_m_640.log 2>&1 &

tail -n 20 picodet_m_640.log

python tools/eval.py -f exps/picodet/picodet_m_640_voc2012.py -d 1 -b 8 -c 16.pth --conf 0.025 --tsize 640

python tools/demo.py image -f exps/picodet/picodet_m_640_voc2012.py -c 16.pth --path assets/000000000019.jpg --conf 0.15 --tsize 640 --save_result --device gpu


实测 picodet_m_640 的AP(0.50:0.95)可以到达0.55+、AP(0.50)可以到达0.75+、AP(small)可以到达0.12+。





----------------------- 恢复训练（加上参数--resume） -----------------------




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


python tools/eval.py -f exps/picodet/picodet_m_416_coco_lcnet.py -d 1 -b 4 -c picodet_m_416_coco_lcnet.pth --conf 0.025 --tsize 416

Average forward time: 7.04 ms, Average NMS time: 0.00 ms, Average inference time: 7.05 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.368
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.525
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.391
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.145
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.400
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.579
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.301
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.475
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.506
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.222
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.571
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.736


python tools/eval.py -f exps/picodet/picodet_l_320_coco_lcnet.py -d 1 -b 4 -c picodet_l_320_coco_lcnet.pth --conf 0.025 --tsize 320

Average forward time: 8.53 ms, Average NMS time: 0.00 ms, Average inference time: 8.53 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.355
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.510
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.377
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.136
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.393
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.569
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.296
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.459
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.488
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.199
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.563
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.739


python tools/eval.py -f exps/picodet/picodet_l_416_coco_lcnet.py -d 1 -b 4 -c picodet_l_416_coco_lcnet.pth --conf 0.025 --tsize 416

Average forward time: 8.58 ms, Average NMS time: 0.00 ms, Average inference time: 8.58 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.389
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.549
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.417
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.167
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.430
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.593
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.315
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.496
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.526
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.252
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.594
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.745


python tools/eval.py -f exps/picodet/picodet_l_640_coco_lcnet.py -d 1 -b 4 -c picodet_l_640_coco_lcnet.pth --conf 0.025 --tsize 640

Average forward time: 9.52 ms, Average NMS time: 0.00 ms, Average inference time: 9.53 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.421
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.588
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.453
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.223
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.460
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.596
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.333
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.533
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.571
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.331
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.628
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.757




----------------------- 导出为ONNX -----------------------




----------------------- 导出为TensorRT -----------------------


----------------------- 复现COCO上的精度 -----------------------
相关训练日志见 train_coco 文件夹

(1)picodet_s_416_coco_lcnet
export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python tools/train.py -f exps/picodet/picodet_s_416_coco_lcnet.py -d 4 -b 192 -eb 16 -w 4 -ew 4 -c PPLCNet_x0_75_pretrained.pth     > picodet_s_416_coco_lcnet_4gpu.log 2>&1 &

训练日志见train_coco/picodet_s_416_coco_lcnet_4gpu.txt
实测训300 epochs后，最高mAP为31.74，基本上能达到转换的官方权重( Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.320)


from scratch:
export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python tools/train.py -f exps/picodet/picodet_s_416_coco_lcnet.py -d 4 -b 192 -eb 16 -w 4 -ew 4     > picodet_s_416_coco_lcnet_from_scratch_4gpu.log 2>&1 &


python tools/eval.py -f exps/picodet/picodet_s_416_coco_lcnet.py -d 1 -b 4 -c 300.pth --conf 0.025 --tsize 416



只有双卡的时候：
export CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/picodet/picodet_s_416_coco_lcnet.py -d 2 -b 96 -eb 16 -w 4 -ew 4 -c PPLCNet_x0_75_pretrained.pth     > picodet_s_416_coco_lcnet_2gpu.log 2>&1 &




