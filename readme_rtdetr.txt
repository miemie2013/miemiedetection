

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


mmdet/models/losses/detr_loss.py
_get_loss_bbox()
可能是boxes被inplace操作，导致无法训练。

encoder_layer(TransformerLayer) 被放进 HybridEncoder 的 self.encoder(nn.ModuleList)
self.encoder里有1个元素，类型是 TransformerEncoder(encoder_layer, num_encoder_layers)


python tools/convert_weights.py -f exps/rtdetr/rtdetr_r18vd_6x_coco.py -c rtdetr_r18vd_dec3_6x_coco.pdparams -oc rtdetr_r18vd_dec3_6x_coco.pth -nc 80 --device gpu

python tools/convert_weights.py -f exps/rtdetr/rtdetr_r50vd_6x_coco.py -c rtdetr_r50vd_6x_coco.pdparams -oc rtdetr_r50vd_6x_coco.pth -nc 80 --device gpu


python tools/convert_weights.py -f exps/rtdetr/rtdetr_r50vd_6x_coco.py -c ResNet50_vd_ssld_v2_pretrained.pdparams -oc ResNet50_vd_ssld_v2_pretrained.pth -nc 80 --only_backbone True




----------------------- 预测 -----------------------
python tools/demo.py image -f exps/rtdetr/rtdetr_r18vd_6x_coco.py -c rtdetr_r18vd_dec3_6x_coco.pth --path assets/000000000019.jpg --conf 0.15 --tsize 640 --save_result --device gpu

python tools/demo.py image -f exps/rtdetr/rtdetr_r50vd_6x_coco.py -c rtdetr_r50vd_6x_coco.pth --path assets/000000000019.jpg --conf 0.15 --tsize 640 --save_result --device gpu








----------------------- 训练 -----------------------




----------------------- 迁移学习，带上-c（--ckpt）参数读取预训练模型。 -----------------------
后台启动：
nohup xxx     > ppyolo.log 2>&1 &


- - - - - - - - - - - - - - - - - - - - - -
python tools/train.py -f exps/rtdetr/rtdetr_r18vd_6x_voc2012.py -d 1 -b 2 -eb 2 -w 0 -ew 0 -c rtdetr_r18vd_dec3_6x_coco.pth



python tools/train.py -f exps/rtdetr/rtdetr_r18vd_6x_coco.py -d 1 -b 2 -eb 2 -w 0 -ew 0 -c rtdetr_r18vd_dec3_6x_coco.pth


python tools/train.py -f exps/rtdetr/rtdetr_r50vd_6x_coco.py -d 1 -b 2 -eb 2 -w 0 -ew 0 -c rtdetr_r50vd_6x_coco.pth




----------------------- 恢复训练（加上参数--resume） -----------------------




----------------------- 评估 -----------------------
python tools/eval.py -f exps/rtdetr/rtdetr_r50vd_6x_coco.py -d 1 -b 8 -w 4 -c rtdetr_r50vd_6x_coco.pth --tsize 640

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.532
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.714
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.578
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.348
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.580
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.701
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.391
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.656
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.722
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.547
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.765
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.880




----------------------- 导出为ONNX -----------------------




----------------------- 导出为TensorRT -----------------------


----------------------- 复现COCO上的精度 -----------------------



