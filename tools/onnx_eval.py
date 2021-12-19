#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os

import cv2
import json
import time
from collections import deque
import numpy as np

import onnxruntime

# add python path of this repo to sys.path
import sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from mmdet.data.data_augment import preproc, preproc_ppyolo
from mmdet.utils import mkdir, multiclass_nms, demo_postprocess, vis, get_classes
from mmdet.utils.demo_utils import numpy_matrix_nms


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--val_image_path",
        type=str,
        default="../COCO/val2017",
        help="Path to val_dataset image.",
    )
    parser.add_argument(
        "-a",
        "--val_anno_path",
        type=str,
        default="../COCO/annotations/instances_val2017.json",
        help="Path to val_annotation json file.",
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        default="eval",
        help="for COCO2017 dataset, select \"eval\" or \"test_dev\"; for custom dataset, only \"eval\".",
    )
    parser.add_argument(
        "-an",
        "--archi_name",
        type=str,
        default='YOLOX',
        help="architecture name.",
    )
    parser.add_argument(
        "-acn",
        "--archi_config_name",
        type=str,
        default='',
        help="architecture config name.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.001,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    return parser


def cocoapi_eval(jsonfile,
                 style,
                 coco_gt=None,
                 anno_file=None,
                 max_dets=(100, 300, 1000)):
    """
    Args:
        jsonfile: Evaluation json file, eg: bbox.json, mask.json.
        style: COCOeval style, can be `bbox` , `segm` and `proposal`.
        coco_gt: Whether to load COCOAPI through anno_file,
                 eg: coco_gt = COCO(anno_file)
        anno_file: COCO annotations file.
        max_dets: COCO evaluation maxDets.
    """
    assert coco_gt != None or anno_file != None
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if coco_gt == None:
        coco_gt = COCO(anno_file)
    print("Start evaluate...")
    coco_dt = coco_gt.loadRes(jsonfile)
    if style == 'proposal':
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.useCats = 0
        coco_eval.params.maxDets = list(max_dets)
    else:
        coco_eval = COCOeval(coco_gt, coco_dt, style)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats


def bbox_eval(anno_file, pred_file):
    from pycocotools.coco import COCO
    coco_gt = COCO(anno_file)
    map_stats = cocoapi_eval(pred_file, 'bbox', coco_gt=coco_gt)
    # flush coco evaluation result
    sys.stdout.flush()
    return map_stats


def mask_eval(anno_file, pred_file):
    from pycocotools.coco import COCO
    coco_gt = COCO(anno_file)
    return cocoapi_eval(pred_file, 'segm', coco_gt=coco_gt)


if __name__ == '__main__':
    args = make_parser().parse_args()
    assert args.eval_type in ["eval", "test_dev"], "for COCO2017 dataset, select \"eval\" or \"test_dev\"; for custom dataset, only \"eval\"."
    # 判断是否是调试状态
    isDebug = True if sys.gettrace() else False
    if isDebug:
        print('Debug Mode.')
        args.model = '../' + args.model
        args.val_image_path = '../' + args.val_image_path   # 如果是绝对路径，把这一行注释掉
        args.val_anno_path = '../' + args.val_anno_path   # 如果是绝对路径，把这一行注释掉


    # 验证集图片的相对路径
    eval_pre_path = args.val_image_path
    anno_file = args.val_anno_path
    from pycocotools.coco import COCO
    val_dataset = COCO(anno_file)
    val_img_ids = val_dataset.getImgIds()
    images = []   # 只跑有gt的图片，跟随PaddleDetection
    for img_id in val_img_ids:
        ins_anno_ids = val_dataset.getAnnIds(imgIds=img_id, iscrowd=False)   # 读取这张图片所有标注anno的id
        if len(ins_anno_ids) == 0:
            continue
        img_anno = val_dataset.loadImgs(img_id)[0]
        images.append(img_anno)
    val_count = len(images)

    # 种类id
    _catid2clsid = {}
    _clsid2catid = {}
    _clsid2cname = {}
    with open(anno_file, 'r', encoding='utf-8') as f2:
        dataset_text = ''
        for line in f2:
            line = line.strip()
            dataset_text += line
        eval_dataset = json.loads(dataset_text)
        categories = eval_dataset['categories']
        for clsid, cate_dic in enumerate(categories):
            catid = cate_dic['id']
            cname = cate_dic['name']
            _catid2clsid[catid] = clsid
            _clsid2catid[clsid] = catid
            _clsid2cname[clsid] = cname
    class_names = []
    num_classes = len(_clsid2cname.keys())
    for clsid in range(num_classes):
        class_names.append(_clsid2cname[clsid])


    input_shape = tuple(map(int, args.input_shape.split(',')))

    if args.archi_name == 'YOLOX':
        # 预处理代码
        img, ratio = preproc(origin_img, input_shape)

        session = onnxruntime.InferenceSession(args.model)

        ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
        output = session.run(None, ort_inputs)
        predictions = demo_postprocess(output[0], input_shape, p6=args.with_p6)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                             conf=args.score_thr, class_names=class_names)
    elif args.archi_name == 'PPYOLO':
        session = onnxruntime.InferenceSession(args.model)
        nms_cfg = dict(
            score_threshold=0.01,
            post_threshold=args.score_thr,
            nms_top_k=500,
            keep_top_k=100,
            use_gaussian=False,
            gaussian_sigma=2.,
        )
        bbox_data = []
        print('Eval Start!')
        start_time = time.time()
        for k, dic in enumerate(images):
            # 预处理代码
            origin_img = cv2.imread(os.path.join(args.val_image_path, dic['file_name']))
            img, im_size = preproc_ppyolo(origin_img, input_shape)

            ort_inputs = {session.get_inputs()[0].name: img, session.get_inputs()[1].name: im_size}
            outputs = session.run(None, ort_inputs)
            output = outputs[0]
            yolo_boxes = output[:, :, :4]   # [N, A,  4]
            yolo_scores = output[:, :, 4:]  # [N, A, 80]

            # nms
            dets = numpy_matrix_nms(yolo_boxes[0, :, :], yolo_scores[0, :, :], **nms_cfg)
            if dets[0][0] > -0.5:
                final_boxes, final_scores, final_cls_inds = dets[:, 2:], dets[:, 1], dets[:, 0]
                # 写入json文件
                im_id = dic['id']
                im_name = dic['file_name']
                n = len(final_boxes)
                for p in range(n):
                    clsid = final_cls_inds[p]
                    score = final_scores[p]
                    xmin, ymin, xmax, ymax = final_boxes[p]
                    catid = (_clsid2catid[int(clsid)])
                    w = xmax - xmin + 1
                    h = ymax - ymin + 1

                    bbox = [xmin, ymin, w, h]
                    # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
                    bbox = [round(float(x) * 10) / 10 for x in bbox]
                    bbox_res = {
                        'image_id': im_id,
                        'category_id': catid,
                        'bbox': bbox,
                        'score': float(score)
                    }
                    bbox_data.append(bbox_res)
            if (k + 1) % 100 == 0:
                print('process %d/%d'%((k + 1), val_count))
        cost = time.time() - start_time
        print('total time: {0:.6f}s'.format(cost))
        print('Speed: %.6fs per image,  %.1f FPS.' % ((cost / val_count), (val_count / cost)))

        # cal mAP
        bbox_path = 'result_bbox.json'
        with open(bbox_path, 'w') as f:
            json.dump(bbox_data, f)
        if args.eval_type == 'eval':
            # 开始评测
            box_ap_stats = bbox_eval(anno_file, bbox_path)
        elif args.eval_type == 'test_dev':
            print('Done.')
    else:
        raise NotImplementedError("Architectures \'{}\' is not implemented.".format(args.archi_name))
