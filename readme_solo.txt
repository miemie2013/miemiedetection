
nvidia-smi


pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


----------------------- 转换权重 -----------------------
wget https://paddledet.bj.bcebos.com/models/solov2_r50_fpn_3x_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/solov2_r50_enhance_coco.pdparams

wget https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_vd_ssld_pretrained.pdparams


python tools/convert_weights.py -f exps/solo/solov2_r50_fpn_3x_coco.py -c solov2_r50_fpn_3x_coco.pdparams -oc solov2_r50_fpn_3x_coco.pth -nc 80

python tools/convert_weights.py -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd_coco.pdparams -oc ppyolo_r18vd.pth -nc 80
python tools/convert_weights.py -f exps/ppyolo/ppyolo_r18vd.py -c ResNet18_vd_pretrained.pdparams -oc ResNet18_vd_pretrained.pth -nc 80 --only_backbone True



----------------------- 预测 -----------------------
python tools/demo.py image -f exps/solo/solov2_r50_fpn_3x_coco.py -c solov2_r50_fpn_3x_coco.pth --path assets/000000000019.jpg --conf 0.15 --tsize 800 --save_result --device gpu





----------------------- 导出为ncnn -----------------------
python tools/demo.py ncnn -f exps/ppyoloe/ppyoloe_crn_s_300e_coco.py -c ppyoloe_crn_s_300e_coco.pth --ncnn_output_path ppyoloe_crn_s_300e_coco

python tools/demo.py ncnn -f exps/ppyoloe/ppyoloe_crn_m_300e_coco.py -c ppyoloe_crn_m_300e_coco.pth --ncnn_output_path ppyoloe_crn_m_300e_coco

python tools/demo.py ncnn -f exps/ppyoloe/ppyoloe_crn_l_300e_coco.py -c ppyoloe_crn_l_300e_coco.pth --ncnn_output_path ppyoloe_crn_l_300e_coco

python tools/demo.py ncnn -f exps/ppyoloe/ppyoloe_crn_x_300e_coco.py -c ppyoloe_crn_x_300e_coco.pth --ncnn_output_path ppyoloe_crn_x_300e_coco

python tools/demo.py ncnn -f exps/ppyoloe/ppyoloe_crn_l_voc2012.py -c PPYOLOE_outputs/ppyoloe_crn_l_voc2012/6.pth --ncnn_output_path ppyoloe_crn_l_voc2012_epoch_6



cd build/examples
./test2_06_ppyoloe_ncnn ../../my_tests/000000000019.jpg ppyoloe_crn_s_300e_coco.param ppyoloe_crn_s_300e_coco.bin

cd build/examples
./test2_06_ppyoloe_ncnn ../../my_tests/000000000019.jpg ppyoloe_crn_m_300e_coco.param ppyoloe_crn_m_300e_coco.bin

cd build/examples
./test2_06_ppyoloe_ncnn ../../my_tests/000000000019.jpg ppyoloe_crn_l_300e_coco.param ppyoloe_crn_l_300e_coco.bin

cd build/examples
./test2_06_ppyoloe_ncnn ../../my_tests/000000000019.jpg ppyoloe_crn_x_300e_coco.param ppyoloe_crn_x_300e_coco.bin

cd build/examples
./test2_06_ppyoloe_ncnn ../../my_tests/000000000019.jpg ppyoloe_crn_l_voc2012_epoch_6.param ppyoloe_crn_l_voc2012_epoch_6.bin



----------------------- 训练 -----------------------
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 8 -eb 4





----------------------- 迁移学习，带上-c（--ckpt）参数读取预训练模型。 -----------------------
后台启动：
nohup xxx     > ppyolo.log 2>&1 &


PPYOLO把RandomShape、NormalizeImage、Permute、Gt2YoloTarget这4个预处理步骤放到了sample_transforms中，
而不是放到batch_transforms中，虽然这样写不美观，但是可以提速n倍。因为用collate_fn实现batch_transforms太耗时了！能不使用batch_transforms尽量不使用batch_transforms！

复现paddle版ppyolo2x迁移学习:（可以加--fp16， -eb表示验证时的批大小）
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_voc2012.py -d 1 -b 8 -eb 4 -c ppyolo_r50vd_2x.pth

python tools/eval.py -f exps/ppyolo/ppyolo_r50vd_voc2012.py -d 1 -b 4 -c PPYOLO_outputs/ppyolo_r50vd_voc2012/16.pth --conf 0.01 --tsize 608


2机2卡训练：(发现一个隐藏知识点：获得损失（训练）、推理 都要放在模型的forward()中进行，否则DDP会计算错误结果。)
CUDA_VISIBLE_DEVICES=0
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_voc2012.py --dist-url tcp://192.168.0.107:12312 --num_machines 2 --machine_rank 0 -b 8 -eb 4 -c ppyolo_r50vd_2x.pth

python tools/eval.py -f exps/ppyolo/ppyolo_r50vd_voc2012.py -d 1 -b 8 -c PPYOLO_outputs/ppyolo_r50vd_voc2012/16.pth --conf 0.01 --tsize 608


1机2卡训练：(发现一个隐藏知识点：获得损失（训练）、推理 都要放在模型的forward()中进行，否则DDP会计算错误结果。)
CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/ppyolo/ppyolo_r50vd_voc2012.py -d 2 -b 8 -eb 2 -c ppyolo_r50vd_2x.pth     > ppyolo.log 2>&1 &

tail -n 20 ppyolo.log


实测ppyolo_r50vd_2x的AP(0.50:0.95)可以到达0.59+、AP(0.50)可以到达0.82+、AP(small)可以到达0.18+。
- - - - - - - - - - - - - - - - - - - - - -


复现paddle版ppyolo_r18vd迁移学习:（可以加--fp16， -eb表示验证时的批大小）
python tools/train.py -f exps/ppyolo/ppyolo_r18vd_voc2012.py -d 1 -b 8 -eb 4 -c ppyolo_r18vd.pth


2机2卡训练：
after_epoch(self):里会调用
            all_reduce_norm(self.model)
多卡训练时会报错，所以设置配置文件里的self.eval_interval = 99999999
CUDA_VISIBLE_DEVICES=0
python tools/train.py -f exps/ppyolo/ppyolo_r18vd_voc2012.py --dist-url tcp://192.168.0.107:12312 --num_machines 2 --machine_rank 0 -b 8 -eb 4 -c ppyolo_r18vd.pth




1机2卡训练：
CUDA_VISIBLE_DEVICES=0,1
python tools/train.py -f exps/ppyolo/ppyolo_r18vd_voc2012.py -d 2 -b 8 -eb 4 -c ppyolo_r18vd.pth


实测ppyolo_r18vd的AP(0.50:0.95)可以到达0.39+、AP(0.50)可以到达0.65+、AP(small)可以到达0.06+。
- - - - - - - - - - - - - - - - - - - - - -

复现paddle版ppyolov2迁移学习:（可以加--fp16， -eb表示验证时的批大小）
python tools/train.py -f exps/ppyolo/ppyolov2_r50vd_voc2012.py -d 1 -b 8 -eb 2 -c ppyolov2_r50vd_365e.pth


2机2卡训练：(发现一个隐藏知识点：获得损失（训练）、推理 都要放在模型的forward()中进行，否则DDP会计算错误结果。)
CUDA_VISIBLE_DEVICES=0
python tools/train.py -f exps/ppyolo/ppyolov2_r50vd_voc2012.py --dist-url tcp://192.168.0.107:12312 --num_machines 2 --machine_rank 0 -b 8 -eb 2 -c ppyolov2_r50vd_365e.pth


1机2卡训练：(发现一个隐藏知识点：获得损失（训练）、推理 都要放在模型的forward()中进行，否则DDP会计算错误结果。)
CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/ppyolo/ppyolov2_r50vd_voc2012.py -d 2 -b 8 -eb 2 -c ppyolov2_r50vd_365e.pth     > ppyolov2.log 2>&1 &

tail -n 20 ppyolov2.log



实测ppyolov2_r50vd_365e的AP(0.50:0.95)可以到达0.63+、AP(0.50)可以到达0.84+、AP(small)可以到达0.25+。


- - - - - - - - - - - - - - - - - - - - - -

复现paddle版ppyoloe_s迁移学习（不冻结骨干网络）:（可以加--fp16， -eb表示验证时的批大小）
python tools/train.py -f exps/ppyoloe/ppyoloe_crn_s_voc2012.py -d 1 -b 4 -eb 2 -c ppyoloe_crn_s_300e_coco.pth --fp16

python tools/eval.py -f exps/ppyoloe/ppyoloe_crn_s_voc2012.py -d 1 -b 4 -c PPYOLOE_outputs/ppyoloe_crn_s_voc2012/16.pth --conf 0.01 --tsize 640

python tools/demo.py image -f exps/ppyoloe/ppyoloe_crn_s_voc2012.py -c PPYOLOE_outputs/ppyoloe_crn_s_voc2012/16.pth --path assets/000000000019.jpg --conf 0.15 --tsize 640 --save_result --device gpu


2机2卡训练：(发现一个隐藏知识点：获得损失（训练）、推理 都要放在模型的forward()中进行，否则DDP会计算错误结果。)
CUDA_VISIBLE_DEVICES=0
python tools/train.py -f exps/ppyoloe/ppyoloe_crn_s_voc2012.py --dist-url tcp://192.168.0.106:12312 --num_machines 2 --machine_rank 0 -b 4 -eb 2 -c ppyoloe_crn_s_300e_coco.pth --fp16


1机2卡训练：(发现一个隐藏知识点：获得损失（训练）、推理 都要放在模型的forward()中进行，否则DDP会计算错误结果。)
CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/ppyoloe/ppyoloe_crn_s_voc2012.py -d 2 -b 4 -eb 2 -c ppyoloe_crn_s_300e_coco.pth --fp16     > ppyoloe_s.log 2>&1 &

tail -n 20 ppyoloe_s.log



实测ppyoloe_s的AP(0.50:0.95)可以到达0.48+、AP(0.50)可以到达0.68+、AP(small)可以到达0.15+。


- - - - - - - - - - - - - - - - - - - - - -

复现paddle版ppyoloe_l迁移学习（冻结了骨干网络）:（可以加--fp16， -eb表示验证时的批大小）
python tools/train.py -f exps/ppyoloe/ppyoloe_crn_l_voc2012.py -d 1 -b 8 -eb 2 -c ppyoloe_crn_l_300e_coco.pth --fp16

python tools/eval.py -f exps/ppyoloe/ppyoloe_crn_l_voc2012.py -d 1 -b 4 -c PPYOLOE_outputs/ppyoloe_crn_l_voc2012/16.pth --conf 0.01 --tsize 640

python tools/demo.py image -f exps/ppyoloe/ppyoloe_crn_l_voc2012.py -c PPYOLOE_outputs/ppyoloe_crn_l_voc2012/16.pth --path assets/000000000019.jpg --conf 0.15 --tsize 640 --save_result --device gpu


1机2卡训练：(发现一个隐藏知识点：获得损失（训练）、推理 都要放在模型的forward()中进行，否则DDP会计算错误结果。)
CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/ppyoloe/ppyoloe_crn_l_voc2012.py -d 2 -b 8 -eb 2 -c ppyoloe_crn_l_300e_coco.pth --fp16     > ppyoloe_l.log 2>&1 &

tail -n 20 ppyoloe_l.log



实测ppyoloe_l的AP(0.50:0.95)可以到达0.66+、AP(0.50)可以到达0.85+、AP(small)可以到达0.28+。


- - - - - - - - - - - - - - - - - - - - - -
感兴趣的同学可以加载 CSPResNetb_l_pretrained.pth 从头训练ppyoloe_l（voc2012数据集）：

python tools/train.py -f exps/ppyoloe/ppyoloe_crn_l_voc2012.py -d 1 -b 8 -eb 2 -c CSPResNetb_l_pretrained.pth --fp16


训练完16个epoch后，实测ppyoloe_l的AP(0.50:0.95)可以到达0.05+、AP(0.50)可以到达0.16+、AP(small)可以到达0.00+。



----------------------- 恢复训练（加上参数--resume） -----------------------
python tools/train.py -f exps/ppyolo/ppyolo_r50vd_2x.py -d 1 -b 8 -eb 4 -c PPYOLO_outputs/ppyolo_r50vd_2x/13.pth --resume


python tools/train.py -f exps/ppyolo/ppyolo_r18vd.py -d 1 -b 16 -eb 8 -c PPYOLO_outputs/ppyolo_r18vd/7.pth --resume





----------------------- 评估 -----------------------
python tools/eval.py -f exps/solo/solov2_r50_fpn_3x_coco.py -d 1 -b 1 -c solov2_r50_fpn_3x_coco.pth --conf 0.01 --tsize 800

python tools/eval.py -f exps/solo/solov2_r50_fpn_3x_coco.py -d 1 -b 2 -c solov2_r50_fpn_3x_coco.pth --conf 0.01 --tsize 800


Average forward time: 34.49 ms, Average NMS time: 0.00 ms, Average inference time: 34.49 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.453
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.654
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.498
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.300
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.485
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.593
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.345
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.578
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.631
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.450
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.666
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.780






----------------------- 导出为ONNX -----------------------
见demo/ONNXRuntime/README.md

会设置model.head.decode_in_inference = False，此时只对置信位和各类别概率进行sigmoid()激活。xywh没有进行解码，更没有进行nms。
python tools/export_onnx.py --output-name ppyolo_r50vd_2x.onnx -f exps/ppyolo/ppyolo_r50vd_2x.py -c ppyolo_r50vd_2x.pth


python tools/export_onnx.py --output-name ppyolov2_r50vd_365e.onnx -f exps/ppyolo/ppyolov2_r50vd_365e.py -c ppyolov2_r50vd_365e.pth


python tools/export_onnx.py --output-name ppyolo_r18vd.onnx -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd.pth



ONNX预测，命令改动为（用numpy对xywh进行解码，进行nms。）


python tools/onnx_inference.py -an PPYOLO -acn ppyolo_r18vd -m ppyolo_r18vd.onnx -i assets/dog.jpg -o ONNX_PPYOLO_R18VD_outputs -s 0.15 --input_shape 416,416 -cn class_names/coco_classes.txt


python tools/onnx_inference.py -an PPYOLO -acn ppyolo_r50vd_2x -m ppyolo_r50vd_2x.onnx -i assets/dog.jpg -o ONNX_PPYOLO_R50VD_outputs -s 0.15 --input_shape 608,608 -cn class_names/coco_classes.txt


python tools/onnx_inference.py -an PPYOLO -acn ppyolov2_r50vd_365e -m ppyolov2_r50vd_365e.onnx -i assets/dog.jpg -o ONNX_PPYOLOv2_R50VD_outputs -s 0.15 --input_shape 640,640 -cn class_names/coco_classes.txt





python tools/onnx_inference.py -an PPYOLO -acn ppyolo_r18vd -m ppyolo_r18vd.onnx -i D://GitHub/Pytorch-YOLO/images/test/000000052996.jpg -o ONNX_PPYOLO_R18VD_outputs -s 0.15 --input_shape 416,416 -cn class_names/coco_classes.txt






用onnx模型进行验证，

python tools/onnx_eval.py -an PPYOLO -acn ppyolo_r18vd -m ppyolo_r18vd.onnx -i ../COCO/val2017 -a ../COCO/annotations/instances_val2017.json -s 0.01 --input_shape 416,416 --eval_type eval





----------------------- 导出为TensorRT -----------------------
python tools_trt/export_trt.py -f exps/ppyolo/ppyolo_r18vd.py -c ppyolo_r18vd.pth --conf 0.15 --tsize 416






