# 如果命令不能成功执行，说明咩酱实现中，，，（and，把PPYOLOv2完全实现再补全文档)
nvidia-smi


pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth


wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth



----------------------- 预处理 -----------------------
原版中YOLOX预处理：
1.读图片
img = cv2.imread(img)
2.填充成正方形，填充值是114,
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
3.保持宽高比把图片最长的边放缩到640，
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
4.transpose操作，HWC变成CHW
padded_img = padded_img.transpose(swap)
5.元素类型由int8转float32
padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
6.如果是旧版本，还需要这样（BGR转RGB，归一化）：
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
7.转Tensor:
img = torch.from_numpy(img).unsqueeze(0)


----------------------- 预测 -----------------------
python tools/demo.py image -f exps/yolox/yolox_s.py -c yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu

python tools/demo.py image -f exps/yolox/yolox_s.py -c yolox_s.pth --path assets/000000000019.jpg --conf 0.15 --tsize 640 --save_result --device gpu


python tools/demo.py image -f exps/yolox/yolox_m.py -c YOLOX_outputs/yolox_m/1.pth --path D://PycharmProjects/Paddle-PPYOLO-master/images/test --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu






----------------------- 评估 -----------------------
python tools/eval.py -f exps/yolox/yolox_s.py -d 1 -b 8 -w 4 -c yolox_s.pth --conf 0.001 --tsize 640


Average forward time: 10.76 ms, Average NMS time: 2.75 ms, Average inference time: 13.51 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.405
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.593
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.438
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.233
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.448
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.541
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.326
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.531
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.574
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.367
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.635
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.724


python tools/eval.py -f exps/yolox/yolox_m.py -d 1 -b 8 -w 4 -c yolox_m.pth --conf 0.001 --tsize 640



----------------------- 复现COCO上的精度 -----------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nohup python tools/train.py -f exps/yolox/yolox_s.py -d 8 -b 64 -eb 64 -w 4 -ew 4 --fp16     > yolox_s_coco.log 2>&1 &

日志见 train_coco/yolox_s_8gpu.txt
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.399
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.588
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.431
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.220
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.443
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.529

python tools/eval.py -f exps/yolox/yolox_s.py -d 1 -b 16 -w 4 -c YOLOX_outputs/yolox_s/300.pth --conf 0.01 --tsize 640

(单卡调试用)
python tools/train.py -f exps/yolox/yolox_s.py -d 1 -b 8 -eb 4 -w 1 -ew 0 --fp16 -c yolox_s.pth



----------------------- 迁移学习，带上-c（--ckpt）参数读取预训练模型。 -----------------------
迁移学习（不冻结骨干网络）:（可以加--fp16， -eb表示验证时的批大小）
python tools/train.py -f exps/yolox/yolox_s_voc2012.py -d 1 -b 24 -eb 16 -w 4 -ew 4 --fp16 -c yolox_s.pth


1机2卡训练：(发现一个隐藏知识点：获得损失（训练）、推理 都要放在模型的forward()中进行，否则DDP会计算错误结果。)
export CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/yolox/yolox_s_voc2012.py -d 2 -b 24 -eb 16 -w 4 -ew 4 --fp16 -c yolox_s.pth     > yolox_s.log 2>&1 &

python tools/eval.py -f exps/yolox/yolox_s_voc2012.py -d 1 -b 8 -w 4 -c 16.pth --conf 0.01 --tsize 640


实测 yolox_s 的AP最高可以到达（日志见 train_ppyolo_in_voc2012/yolox_s_voc2012.txt ）
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.509
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.745
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.570
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.194
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.374
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.583


- - - - - - - - - - - -
export CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/yolox/yolox_s_simple_voc2012.py -d 2 -b 24 -eb 16 -w 4 -ew 4 --fp16 -c yolox_s.pth     > yolox_s_simple.log 2>&1 &

(单卡调试用)
python tools/train.py -f exps/yolox/yolox_s_simple_voc2012.py -d 1 -b 8 -eb 4 -w 1 -ew 0 --fp16 -c yolox_s.pth


python tools/eval.py -f exps/yolox/yolox_s_simple_voc2012.py -d 1 -b 8 -w 4 -c 16.pth --conf 0.01 --tsize 640

python tools/demo.py image -f exps/yolox/yolox_s_simple_voc2012.py -c 16.pth --path assets/2008_000073.jpg --conf 0.15 --tsize 640 --save_result --device gpu


实测 yolox_s_simple 的AP最后可以到达（日志见 train_ppyolo_in_voc2012/yolox_s_simple_voc2012.txt ）
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.408
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.629
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.451
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.102
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.268
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.488
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.372
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.545
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.556
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.217
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.429
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.630

导出给私有仓库：
python tools/convert_weights.py -f exps/yolox/yolox_s_simple_voc2012.py -c YOLOX_outputs/yolox_s_simple_voc2012/16.pth -oc new_16.pth -nc 20 -pp0


----------------------- 恢复训练（加上参数--resume） -----------------------
export CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/yolox/yolox_s_simple_voc2012.py -d 2 -b 24 -eb 16 -w 4 -ew 4 --fp16 -c 10.pth --resume     > yolox_s_simple.log 2>&1 &



----------------------- 导出为ONNX -----------------------
见demo/ONNXRuntime/README.md

会设置model.head.decode_in_inference = False，此时只对置信位和各类别概率进行sigmoid()激活。xywh没有进行解码，更没有进行nms。
python tools/export_onnx.py --output-name yolox_s.onnx -f exps/yolox/yolox_s.py -c yolox_s.pth



ONNX预测，命令改动为（用numpy对xywh进行解码，进行nms。）
python tools/onnx_inference.py -m yolox_s.onnx -i assets/dog.jpg -o ONNX_YOLOX_outputs -s 0.3 --input_shape 640,640 -cn ./class_names/coco_classes.txt



用onnx模型进行验证，

python tools/onnx_eval.py -an YOLOX -m yolox_s.onnx -i ../COCO/val2017 -a ../COCO/annotations/instances_val2017.json -s 0.001 --input_shape 640,640 --eval_type eval

(不是因为少预测了没有gt的图片)
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.394
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.578
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.426
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.217
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.437
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.525
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.311
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.497
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.535
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.327
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.588
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.670

