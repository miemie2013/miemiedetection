# 如果命令不能成功执行，说明咩酱实现中，，，（and，把PPYOLOv2完全实现再补全文档)
nvidia-smi


pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


----------------------- 转换权重 -----------------------
wget https://cloudstor.aarnet.edu.au/plus/s/TlnlXUr6lNNSyoZ/download
wget https://paddledet.bj.bcebos.com/models/fcos_r50_fpn_multiscale_2x_coco.pdparams



python tools/convert_weights.py -f exps/fcos/fcos_rt_r50_fpn_4x.py -c FCOS_RT_MS_R_50_4x_syncbn.pth -oc fcos_rt_r50_syncbn_fpn_4x.pth -nc 80


python tools/convert_weights.py -f exps/fcos/fcos_r50_fpn_2x.py -c fcos_r50_fpn_multiscale_2x.pdparams -oc fcos_r50_fpn_2x.pth -nc 80




----------------------- 预测 -----------------------
python tools/demo.py image -f exps/fcos/fcos_rt_r50_fpn_4x.py -c fcos_rt_r50_syncbn_fpn_4x.pth --path assets/000000000019.jpg --conf 0.15 --tsize 512 --save_result --device gpu


python tools/demo.py image -f exps/fcos/fcos_r50_fpn_2x.py -c fcos_r50_fpn_2x.pth --path assets/000000000019.jpg --conf 0.15 --tsize 800 --save_result --device gpu







----------------------- 评估 -----------------------
python tools/eval.py -f exps/fcos/fcos_rt_r50_fpn_4x.py -d 1 -b 2 -c fcos_rt_r50_syncbn_fpn_4x.pth --conf 0.01 --tsize 512


Average forward time: 40.01 ms, Average NMS time: 0.01 ms, Average inference time: 40.02 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.400
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.585
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.433
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.228
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.441
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.553
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.328
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.537
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.584
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.354
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.640
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.772


python tools/eval.py -f exps/fcos/fcos_r50_fpn_2x.py -d 1 -b 1 -c fcos_r50_fpn_2x.pth --conf 0.01 --tsize 800


Average forward time: 82.21 ms, Average NMS time: 0.01 ms, Average inference time: 82.22 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.406
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.602
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.440
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.232
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.446
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.535
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.331
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.546
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.594
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.385
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.643
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.768





----------------------- 训练 -----------------------
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 8 --fp16





----------------------- 迁移学习，带上-c（--ckpt）参数读取预训练模型。 -----------------------
复现paddle版ppyolo2x迁移学习:（可以加--fp16， -eb表示验证时的批大小）
python tools/train.py -f exps/fcos/fcos_rt_r50_fpn_4x.py -d 1 -b 6 -eb 1 -c fcos_rt_r50_syncbn_fpn_4x.pth


实测fcos_rt_r50_fpn_4x的AP(0.50:0.95)可以到达0.49+、AP(0.50)可以到达0.71+、AP(small)可以到达0.19+。


python tools/train.py -f exps/fcos/fcos_r50_fpn_2x.py -d 1 -b 4 -eb 1 -c fcos_r50_fpn_2x.pth


实测fcos_r50_fpn_2x的AP(0.50:0.95)可以到达0.xx+、AP(0.50)可以到达0.xx+、AP(small)可以到达0.xx+。



----------------------- 恢复训练（加上参数--resume） -----------------------
python tools/train.py -f exps/fcos/fcos_rt_r50_fpn_4x.py -d 1 -b 6 -eb 1 -c FCOS_outputs/fcos_rt_r50_fpn_4x/3.pth --resume





----------------------- 导出为ONNX -----------------------
见demo/ONNXRuntime/README.md

会设置model.head.decode_in_inference = False，此时只对置信位和各类别概率进行sigmoid()激活。xywh没有进行解码，更没有进行nms。
python tools/export_onnx.py --output-name fcos_rt_r50_syncbn_fpn_4x.onnx -f exps/fcos/fcos_rt_r50_fpn_4x.py -c fcos_rt_r50_syncbn_fpn_4x.pth


python tools/export_onnx.py --output-name ppyolo_r18vd.onnx -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd.pth



ONNX预测，命令改动为（用numpy对xywh进行解码，进行nms。）


python tools/onnx_inference.py -an PPYOLO -acn ppyolo_r18vd -m ppyolo_r18vd.onnx -i assets/dog.jpg -o ONNX_PPYOLO_R18VD_outputs -s 0.15 --input_shape 416,416 -cn class_names/coco_classes.txt


python tools/onnx_inference.py -an PPYOLO -acn ppyolo_r18vd -m ppyolo_r18vd.onnx -i D://GitHub/Pytorch-YOLO/images/test/000000052996.jpg -o ONNX_PPYOLO_R18VD_outputs -s 0.15 --input_shape 416,416 -cn class_names/coco_classes.txt






用onnx模型进行验证，

python tools/onnx_eval.py -an PPYOLO -acn ppyolo_r18vd -m ppyolo_r18vd.onnx -i ../COCO/val2017 -a ../COCO/annotations/instances_val2017.json -s 0.01 --input_shape 416,416 --eval_type eval


python tools/onnx_eval.py -an PPYOLO -acn ppyolo_r18vd -m ppyolo_r18vd.onnx -i ../COCO/val2017 -a ../COCO/annotations/instances_val2017.json -s 0.01 --input_shape 416,416 --eval_type test_dev



