
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


----------------------- 转换权重 -----------------------
wget https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_300e_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/ppyoloe_crn_m_300e_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/ppyoloe_crn_x_300e_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/pretrained/CSPResNetb_s_pretrained.pdparams
wget https://paddledet.bj.bcebos.com/models/pretrained/CSPResNetb_m_pretrained.pdparams
wget https://paddledet.bj.bcebos.com/models/pretrained/CSPResNetb_l_pretrained.pdparams
wget https://paddledet.bj.bcebos.com/models/pretrained/CSPResNetb_x_pretrained.pdparams


wget https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_s_80e_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_m_80e_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_l_80e_coco.pdparams
wget https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_x_80e_coco.pdparams
wget https://bj.bcebos.com/v1/paddledet/models/pretrained/ppyoloe_crn_s_obj365_pretrained.pdparams
wget https://bj.bcebos.com/v1/paddledet/models/pretrained/ppyoloe_crn_m_obj365_pretrained.pdparams
wget https://bj.bcebos.com/v1/paddledet/models/pretrained/ppyoloe_crn_l_obj365_pretrained.pdparams
wget https://bj.bcebos.com/v1/paddledet/models/pretrained/ppyoloe_crn_x_obj365_pretrained.pdparams



python tools/convert_weights.py -f exps/ppyoloe/ppyoloe_crn_s_300e_coco.py -c ppyoloe_crn_s_300e_coco.pdparams -oc ppyoloe_crn_s_300e_coco.pth -nc 80
python tools/convert_weights.py -f exps/ppyoloe/ppyoloe_crn_m_300e_coco.py -c ppyoloe_crn_m_300e_coco.pdparams -oc ppyoloe_crn_m_300e_coco.pth -nc 80
python tools/convert_weights.py -f exps/ppyoloe/ppyoloe_crn_l_300e_coco.py -c ppyoloe_crn_l_300e_coco.pdparams -oc ppyoloe_crn_l_300e_coco.pth -nc 80
python tools/convert_weights.py -f exps/ppyoloe/ppyoloe_crn_x_300e_coco.py -c ppyoloe_crn_x_300e_coco.pdparams -oc ppyoloe_crn_x_300e_coco.pth -nc 80

python tools/convert_weights.py -f exps/ppyoloe/ppyoloe_crn_s_300e_coco.py -c CSPResNetb_s_pretrained.pdparams -oc CSPResNetb_s_pretrained.pth -nc 80 --only_backbone True
python tools/convert_weights.py -f exps/ppyoloe/ppyoloe_crn_m_300e_coco.py -c CSPResNetb_m_pretrained.pdparams -oc CSPResNetb_m_pretrained.pth -nc 80 --only_backbone True
python tools/convert_weights.py -f exps/ppyoloe/ppyoloe_crn_l_300e_coco.py -c CSPResNetb_l_pretrained.pdparams -oc CSPResNetb_l_pretrained.pth -nc 80 --only_backbone True
python tools/convert_weights.py -f exps/ppyoloe/ppyoloe_crn_x_300e_coco.py -c CSPResNetb_x_pretrained.pdparams -oc CSPResNetb_x_pretrained.pth -nc 80 --only_backbone True


python tools/convert_weights.py -f exps/ppyoloe_plus/ppyoloe_plus_crn_s_80e_coco.py -c ppyoloe_plus_crn_s_80e_coco.pdparams -oc ppyoloe_plus_crn_s_80e_coco.pth -nc 80
python tools/convert_weights.py -f exps/ppyoloe_plus/ppyoloe_plus_crn_m_80e_coco.py -c ppyoloe_plus_crn_m_80e_coco.pdparams -oc ppyoloe_plus_crn_m_80e_coco.pth -nc 80
python tools/convert_weights.py -f exps/ppyoloe_plus/ppyoloe_plus_crn_l_80e_coco.py -c ppyoloe_plus_crn_l_80e_coco.pdparams -oc ppyoloe_plus_crn_l_80e_coco.pth -nc 80
python tools/convert_weights.py -f exps/ppyoloe_plus/ppyoloe_plus_crn_x_80e_coco.py -c ppyoloe_plus_crn_x_80e_coco.pdparams -oc ppyoloe_plus_crn_x_80e_coco.pth -nc 80

python tools/convert_weights.py -f exps/ppyoloe_plus/ppyoloe_plus_crn_s_80e_coco.py -c ppyoloe_crn_s_obj365_pretrained.pdparams -oc ppyoloe_crn_s_obj365_pretrained.pth -nc 365
python tools/convert_weights.py -f exps/ppyoloe_plus/ppyoloe_plus_crn_m_80e_coco.py -c ppyoloe_crn_m_obj365_pretrained.pdparams -oc ppyoloe_crn_m_obj365_pretrained.pth -nc 365
python tools/convert_weights.py -f exps/ppyoloe_plus/ppyoloe_plus_crn_l_80e_coco.py -c ppyoloe_crn_l_obj365_pretrained.pdparams -oc ppyoloe_crn_l_obj365_pretrained.pth -nc 365
python tools/convert_weights.py -f exps/ppyoloe_plus/ppyoloe_plus_crn_x_80e_coco.py -c ppyoloe_crn_x_obj365_pretrained.pdparams -oc ppyoloe_crn_x_obj365_pretrained.pth -nc 365


----------------------- 预测 -----------------------

python tools/demo.py image -f exps/ppyoloe/ppyoloe_crn_s_300e_coco.py -c ppyoloe_crn_s_300e_coco.pth --path assets/000000000019.jpg --conf 0.15 --tsize 640 --save_result --device gpu

python tools/demo.py image -f exps/ppyoloe/ppyoloe_crn_m_300e_coco.py -c ppyoloe_crn_m_300e_coco.pth --path assets/000000000019.jpg --conf 0.15 --tsize 640 --save_result --device gpu

python tools/demo.py image -f exps/ppyoloe/ppyoloe_crn_l_300e_coco.py -c ppyoloe_crn_l_300e_coco.pth --path assets/000000000019.jpg --conf 0.15 --tsize 640 --save_result --device gpu

python tools/demo.py image -f exps/ppyoloe/ppyoloe_crn_x_300e_coco.py -c ppyoloe_crn_x_300e_coco.pth --path assets/000000000019.jpg --conf 0.15 --tsize 640 --save_result --device gpu

python tools/demo.py image -f exps/ppyoloe/ppyoloe_crn_l_voc2012.py -c 6.pth --path assets/000000000019.jpg --conf 0.15 --tsize 640 --save_result --device gpu

python tools/demo.py image -f exps/ppyoloe_plus/ppyoloe_plus_crn_s_80e_coco.py -c ppyoloe_plus_crn_s_80e_coco.pth --path assets/000000000019.jpg --conf 0.15 --tsize 640 --save_result --device gpu




----------------------- 迁移学习，带上-c（--ckpt）参数读取预训练模型。 -----------------------
后台启动：
nohup xxx     > ppyolo.log 2>&1 &


复现paddle版ppyoloe_s迁移学习（不冻结骨干网络）:（可以加--fp16， -eb表示验证时的批大小）
python tools/train.py -f exps/ppyoloe/ppyoloe_crn_s_voc2012.py -d 1 -b 4 -eb 2 -c ppyoloe_crn_s_300e_coco.pth --fp16

python tools/eval.py -f exps/ppyoloe/ppyoloe_crn_s_voc2012.py -d 1 -b 4 -c 16.pth --conf 0.01 --tsize 640

python tools/demo.py image -f exps/ppyoloe/ppyoloe_crn_s_voc2012.py -c 16.pth --path assets/000000000019.jpg --conf 0.15 --tsize 640 --save_result --device gpu


2机2卡训练：(发现一个隐藏知识点：获得损失（训练）、推理 都要放在模型的forward()中进行，否则DDP会计算错误结果。)
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -f exps/ppyoloe/ppyoloe_crn_s_voc2012.py --dist-url tcp://192.168.0.106:12312 --num_machines 2 --machine_rank 0 -b 4 -eb 2 -c ppyoloe_crn_s_300e_coco.pth --fp16


1机2卡训练：(发现一个隐藏知识点：获得损失（训练）、推理 都要放在模型的forward()中进行，否则DDP会计算错误结果。)
export CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/ppyoloe/ppyoloe_crn_s_voc2012.py -d 2 -b 24 -eb 24 -w 4 -ew 4 -lrs 0.7 -c ppyoloe_crn_s_300e_coco.pth     > ppyoloe_s.log 2>&1 &

tail -n 20 ppyoloe_s.log



实测ppyoloe_s的AP(0.50:0.95)可以到达0.48+、AP(0.50)可以到达0.68+、AP(small)可以到达0.15+。


- - - - - - - - - - - - - - - - - - - - - -

训练 ppyoloe_xs

python tools/train.py -f exps/ppyoloe/ppyoloe_crn_xs_voc2012.py -d 1 -b 4 -eb 2 -c ppyoloe_crn_s_300e_coco.pth




实测ppyoloe_xs的AP(0.50:0.95)可以到达0.xx+、AP(0.50)可以到达0.xx+、AP(small)可以到达0.xx+。


- - - - - - - - - - - - - - - - - - - - - -

复现paddle版ppyoloe_l迁移学习（冻结了骨干网络）:（可以加--fp16， -eb表示验证时的批大小）
python tools/train.py -f exps/ppyoloe/ppyoloe_crn_l_voc2012.py -d 1 -b 8 -eb 2 -c ppyoloe_crn_l_300e_coco.pth --fp16

python tools/eval.py -f exps/ppyoloe/ppyoloe_crn_l_voc2012.py -d 1 -b 4 -c 16.pth --conf 0.01 --tsize 640

python tools/demo.py image -f exps/ppyoloe/ppyoloe_crn_l_voc2012.py -c 16.pth --path assets/000000000019.jpg --conf 0.15 --tsize 640 --save_result --device gpu


1机2卡训练：(发现一个隐藏知识点：获得损失（训练）、推理 都要放在模型的forward()中进行，否则DDP会计算错误结果。)
export CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/ppyoloe/ppyoloe_crn_l_voc2012.py -d 2 -b 8 -eb 2 -c ppyoloe_crn_l_300e_coco.pth --fp16     > ppyoloe_l.log 2>&1 &

tail -n 20 ppyoloe_l.log



实测ppyoloe_l的AP(0.50:0.95)可以到达0.66+、AP(0.50)可以到达0.85+、AP(small)可以到达0.28+。


- - - - - - - - - - - - - - - - - - - - - -
感兴趣的同学可以加载 CSPResNetb_l_pretrained.pth 从头训练ppyoloe_l（voc2012数据集）：

python tools/train.py -f exps/ppyoloe/ppyoloe_crn_l_voc2012.py -d 1 -b 8 -eb 2 -c CSPResNetb_l_pretrained.pth --fp16


训练完16个epoch后，实测ppyoloe_l的AP(0.50:0.95)可以到达0.05+、AP(0.50)可以到达0.16+、AP(small)可以到达0.00+。



- - - - - - - - - - - - - - - - - - - - - -
ppyoloe_plus_s迁移学习（不冻结骨干网络）:（可以加--fp16， -eb表示验证时的批大小）

python tools/eval.py -f exps/ppyoloe_plus/ppyoloe_plus_crn_s_voc2012.py -d 1 -b 4 -c 16.pth --conf 0.01 --tsize 640

python tools/demo.py image -f exps/ppyoloe_plus/ppyoloe_plus_crn_s_voc2012.py -c 16.pth --path assets/000000000019.jpg --conf 0.15 --tsize 640 --save_result --device gpu


1机2卡训练：(发现一个隐藏知识点：获得损失（训练）、推理 都要放在模型的forward()中进行，否则DDP会计算错误结果。)
export CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/ppyoloe_plus/ppyoloe_plus_crn_s_voc2012.py -d 2 -b 16 -eb 8 -w 4 -ew 4 -lrs 1.0 -c ppyoloe_crn_s_obj365_pretrained.pth --fp16     > ppyoloe_plus_s_from_obj365.log 2>&1 &

tail -n 20 ppyoloe_plus_s_from_obj365.log


实测 ppyoloe_plus_s_from_obj365 的AP(0.50:0.95)可以到达0.59+、AP(0.50)可以到达0.78+、AP(small)可以到达0.20+。
日志见 train_ppyolo_in_voc2012/mmdet_ppyoloe_plus_s_from_obj365_2gpu.txt

- - - - - - -
读 COCO 预训练模型（实际上由obj365 fine-tune 得到）进行fine-tune：
export CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/ppyoloe_plus/ppyoloe_plus_crn_s_voc2012.py -d 2 -b 16 -eb 8 -w 4 -ew 4 -lrs 1.0 -c ppyoloe_plus_crn_s_80e_coco.pth     > ppyoloe_plus_s_from_coco.log 2>&1 &

实测 ppyoloe_plus_s_from_coco 的AP(0.50:0.95)可以到达0.62+、AP(0.50)可以到达0.81+、AP(small)可以到达0.24+。
日志见 train_ppyolo_in_voc2012/mmdet_ppyoloe_plus_s_from_obj365(to_coco)_2gpu.txt


- - - - - - -
做 ppyoloe_s 的对比实验（读 ppyoloe 的模型）：
export CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/ppyoloe_plus/ppyoloe_plus_crn_s_voc2012.py -d 2 -b 16 -eb 8 -w 4 -ew 4 -lrs 1.0 -c ppyoloe_crn_s_300e_coco.pth --fp16     > ppyoloe_s_from_coco.log 2>&1 &

实测 ppyoloe_s_from_coco 的AP(0.50:0.95)可以到达0.61+、AP(0.50)可以到达0.80+、AP(small)可以到达0.23+。
日志见 train_ppyolo_in_voc2012/mmdet_ppyoloe_plus_s_from_coco_2gpu.txt



- - - - - - - - - - - - - - - - - - - - - -
pcp:
export CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/ppyoloe_plus/ppyoloe_plus_crn_s_pcp.py -d 2 -b 24 -eb 8 -c ppyoloe_crn_s_obj365_pretrained.pth     > ppyoloe_plus_s_from_obj365.log 2>&1 &

实测 ppyoloe_plus_s_from_obj365 的AP(0.50:0.95)可以到达0.xx+、AP(0.50)可以到达0.xx+、AP(small)可以到达0.xx+。
日志见 train_ppyolo_in_voc2012/xxxxxxxxxx.txt

- - - - - - -
读 COCO 预训练模型（实际上由obj365 fine-tune 得到）进行fine-tune：
export CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/ppyoloe_plus/ppyoloe_plus_crn_s_pcp.py -d 2 -b 24 -eb 8 -c ppyoloe_plus_crn_s_80e_coco.pth     > ppyoloe_plus_s_from_coco.log 2>&1 &

实测 ppyoloe_plus_s_from_coco 的AP(0.50:0.95)可以到达0.xx+、AP(0.50)可以到达0.xx+、AP(small)可以到达0.xx+。
日志见 train_ppyolo_in_voc2012/xxxxxxxxxx.txt


- - - - - - -
做 ppyoloe_s 的对比实验（读 ppyoloe 的模型）：
export CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/ppyoloe_plus/ppyoloe_plus_crn_s_pcp.py -d 2 -b 24 -eb 8 -c ppyoloe_crn_s_300e_coco.pth     > ppyoloe_s_from_coco.log 2>&1 &

实测 ppyoloe_s_from_coco 的AP(0.50:0.95)可以到达0.xx+、AP(0.50)可以到达0.xx+、AP(small)可以到达0.xx+。
日志见 train_ppyolo_in_voc2012/xxxxxxxxxx.txt



----------------------- 知识蒸馏 -----------------------
蒸馏损失有 F.binary_cross_entropy, 所以不使用 fp16, 除非 F.binary_cross_entropy 前转成float32 且 使用 with torch.cuda.amp.autocast(enabled=False):
1.先训练老师模型:
读 COCO 预训练模型（实际上由obj365 fine-tune 得到）进行fine-tune：
export CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/ppyoloe_plus/ppyoloe_plus_crn_l_voc2012.py -d 2 -b 16 -eb 8 -w 4 -ew 4 -lrs 1.0 -c ppyoloe_plus_crn_l_80e_coco.pth     > ppyoloe_plus_l_from_coco.log 2>&1 &

实测 ppyoloe_plus_l_from_coco 的AP(0.50:0.95)可以到达0.xxx+、AP(0.50)可以到达0.xxx+、AP(small)可以到达0.xxx+。
日志见 train_ppyolo_in_voc2012/ppyoloe_plus_l_from_obj365(to_coco)_2gpu.txt


2.蒸馏:
读 COCO 预训练模型（实际上由obj365 fine-tune 得到）进行fine-tune：
export CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/ppyoloe_plus/ppyoloe_plus_crn_s_voc2012.py -d 2 -b 16 -eb 8 -w 4 -ew 4 -lrs 1.0 -c ppyoloe_plus_crn_s_80e_coco.pth -sf exps/slim/distill/ppyoloe_plus_crn_l_voc2012_l2s.py -sc PPYOLOEPlus_outputs/ppyoloe_plus_crn_l_voc2012/1.pth     > ppyoloe_plus_s_from_coco.log 2>&1 &

(单卡调试)
python tools/train.py -f exps/ppyoloe_plus/ppyoloe_plus_crn_s_voc2012.py -d 1 -b 4 -eb 4 -w 1 -ew 1 -lrs 1.0 -c ppyoloe_plus_crn_s_80e_coco.pth -sf exps/slim/distill/ppyoloe_plus_crn_l_voc2012_l2s.py -sc PPYOLOEPlus_outputs/ppyoloe_plus_crn_l_voc2012/1.pth


实测 ppyoloe_plus_s_from_coco 的AP(0.50:0.95)可以到达0.62+、AP(0.50)可以到达0.81+、AP(small)可以到达0.24+。
日志见 train_ppyolo_in_voc2012/mmdet_ppyoloe_plus_s_from_obj365(to_coco)_2gpu.txt



仔细检查：
单卡和多卡时，PPdetModelEMA 的 state_dict 不保存老师的权重，训练时ema也不更新老师的权重
单卡和多卡训练时时，老师的权重不被训练，bn的均值和方差也不会变化
evaluate_and_save_model()
aaaaaaaa.py 对比查看 老师权重的变化



----------------------- 恢复训练（加上参数--resume） -----------------------


----------------------- 评估 -----------------------

python tools/eval.py -f exps/ppyoloe/ppyoloe_crn_s_300e_coco.py -d 1 -b 8 -w 4 -c ppyoloe_crn_s_300e_coco.pth --conf 0.01 --tsize 640

 Average forward time: 21.56 ms, Average NMS time: 0.00 ms, Average inference time: 21.56 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.423
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.593
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.458
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.237
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.464
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.582
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.336
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.546
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.583
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.356
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.640
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.756


python tools/eval.py -f exps/ppyoloe/ppyoloe_crn_s_300e_coco.py -d 1 -b 8 -w 4 -c ppyoloe_crn_s_300e_coco.pth --conf 0.01 --tsize 320

Average forward time: 14.12 ms, Average NMS time: 0.00 ms, Average inference time: 14.12 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.341
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.491
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.362
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.117
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.378
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.571
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.288
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.450
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.477
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.174
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.549
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.744


python tools/eval.py -f exps/ppyoloe/ppyoloe_crn_m_300e_coco.py -d 1 -b 8 -c ppyoloe_crn_m_300e_coco.pth --conf 0.01 --tsize 640

Average forward time: 35.90 ms, Average NMS time: 0.00 ms, Average inference time: 35.90 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.482
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.654
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.523
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.294
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.527
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.645
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.365
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.597
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.635
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.427
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.687
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.803



python tools/eval.py -f exps/ppyoloe/ppyoloe_crn_l_300e_coco.py -d 1 -b 4 -c ppyoloe_crn_l_300e_coco.pth --conf 0.01 --tsize 640

Average forward time: 49.45 ms, Average NMS time: 0.00 ms, Average inference time: 49.45 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.505
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.681
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.547
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.335
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.552
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.671
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.377
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.617
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.657
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.461
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.709
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.819


python tools/eval.py -f exps/ppyoloe/ppyoloe_crn_x_300e_coco.py -d 1 -b 2 -c ppyoloe_crn_x_300e_coco.pth --conf 0.01 --tsize 640

Average forward time: 79.69 ms, Average NMS time: 0.00 ms, Average inference time: 79.70 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.515
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.690
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.555
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.338
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.562
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.682
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.383
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.628
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.669
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.484
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.718
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.823



python tools/eval.py -f exps/ppyoloe_plus/ppyoloe_plus_crn_s_80e_coco.py -d 1 -b 8 -c ppyoloe_plus_crn_s_80e_coco.pth --conf 0.01 --tsize 640

Average forward time: 15.57 ms, Average NMS time: 0.00 ms, Average inference time: 15.57 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.431
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.599
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.474
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.256
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.469
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.583
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.342
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.567
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.625
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.424
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.688
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.777


python tools/eval.py -f exps/ppyoloe_plus/ppyoloe_plus_crn_m_80e_coco.py -d 1 -b 8 -c ppyoloe_plus_crn_m_80e_coco.pth --conf 0.01 --tsize 640

Average forward time: 20.78 ms, Average NMS time: 0.00 ms, Average inference time: 20.78 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.492
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.663
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.539
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.306
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.534
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.656
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.373
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.614
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.673
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.491
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.728
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.823



python tools/eval.py -f exps/ppyoloe_plus/ppyoloe_plus_crn_l_80e_coco.py -d 1 -b 8 -c ppyoloe_plus_crn_l_80e_coco.pth --conf 0.01 --tsize 640

Average forward time: 22.93 ms, Average NMS time: 0.00 ms, Average inference time: 22.93 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.523
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.694
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.573
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.344
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.571
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.685
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.385
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.641
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.699
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.521
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.755
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.846


python tools/eval.py -f exps/ppyoloe_plus/ppyoloe_plus_crn_x_80e_coco.py -d 1 -b 8 -c ppyoloe_plus_crn_x_80e_coco.pth --conf 0.01 --tsize 640

Average forward time: 27.18 ms, Average NMS time: 0.00 ms, Average inference time: 27.18 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.542
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.712
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.594
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.372
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.587
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.699
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.394
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.657
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.711
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.542
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.765
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.851



----------------------- 复现COCO上的精度 -----------------------

(5)ppyoloe_crn_s_300e_coco
(
PaddleDetection-release-2.4 的总批大小是 256=8*32
PaddleDetection-release-2.6 的总批大小是 64=8*8
另外，加上--fp16实测会导致loss出现nan，所以不加，不使用混合精度训练。
)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nohup python tools/train.py -f exps/ppyoloe/ppyoloe_crn_s_300e_coco.py -d 8 -b 64 -eb 64 -w 4 -ew 4 -c CSPResNetb_s_pretrained.pth     > ppyoloe_crn_s_300e_coco_8gpu.log 2>&1 &

训练日志见 train_coco/ppyoloe_s_8gpu.txt
实测训300 epochs后，最高mAP为42.10，基本上能达到转换的官方权重( Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.423)

python tools/eval.py -f exps/ppyoloe/ppyoloe_crn_s_300e_coco.py -d 8 -b 64 -c 300.pth --conf 0.01 --tsize 640

只有双卡的时候：
export CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/ppyoloe/ppyoloe_crn_s_300e_coco.py -d 2 -b 16 -eb 16 -w 4 -ew 4 -c CSPResNetb_s_pretrained.pth     > ppyoloe_crn_s_300e_coco_2gpu.log 2>&1 &




----------------------- 导出为ncnn -----------------------
python tools/demo.py ncnn -f exps/ppyoloe/ppyoloe_crn_s_300e_coco.py -c ppyoloe_crn_s_300e_coco.pth --ncnn_output_path ppyoloe_crn_s_300e_coco

python tools/demo.py ncnn -f exps/ppyoloe/ppyoloe_crn_m_300e_coco.py -c ppyoloe_crn_m_300e_coco.pth --ncnn_output_path ppyoloe_crn_m_300e_coco

python tools/demo.py ncnn -f exps/ppyoloe/ppyoloe_crn_l_300e_coco.py -c ppyoloe_crn_l_300e_coco.pth --ncnn_output_path ppyoloe_crn_l_300e_coco

python tools/demo.py ncnn -f exps/ppyoloe/ppyoloe_crn_x_300e_coco.py -c ppyoloe_crn_x_300e_coco.pth --ncnn_output_path ppyoloe_crn_x_300e_coco

python tools/demo.py ncnn -f exps/ppyoloe/ppyoloe_crn_l_voc2012.py -c 6.pth --ncnn_output_path ppyoloe_crn_l_voc2012_epoch_6



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


./test2_06_ppyoloe_ncnn ../../my_tests/000000013659.jpg ppyoloe_crn_s_300e_coco.param ppyoloe_crn_s_300e_coco.bin

./test2_06_ppyoloe_ncnn ../../my_tests/000000013659.jpg ppyoloe_crn_m_300e_coco.param ppyoloe_crn_m_300e_coco.bin

./test2_06_ppyoloe_ncnn ../../my_tests/000000013659.jpg ppyoloe_crn_l_300e_coco.param ppyoloe_crn_l_300e_coco.bin

./test2_06_ppyoloe_ncnn ../../my_tests/000000013659.jpg ppyoloe_crn_x_300e_coco.param ppyoloe_crn_x_300e_coco.bin

./test2_06_ppyoloe_ncnn ../../my_tests/000000013659.jpg ppyoloe_crn_l_voc2012_epoch_6.param ppyoloe_crn_l_voc2012_epoch_6.bin








