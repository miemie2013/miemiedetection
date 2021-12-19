#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os

import cv2
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
        "--image_path",
        type=str,
        default='test_image.png',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-cn",
        "--cls_names",
        type=str,
        default='./class_names/coco_classes.txt',
        help="Path to class names.",
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
        "-o",
        "--output_dir",
        type=str,
        default='demo_output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
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


if __name__ == '__main__':
    args = make_parser().parse_args()
    # 判断是否是调试状态
    isDebug = True if sys.gettrace() else False
    if isDebug:
        print('Debug Mode.')
        args.model = '../' + args.model
        args.image_path = '../' + args.image_path   # 如果是绝对路径，把这一行注释掉
        args.output_dir = '../' + args.output_dir
        args.cls_names = '../' + args.cls_names


    input_shape = tuple(map(int, args.input_shape.split(',')))
    origin_img = cv2.imread(args.image_path)
    class_names = get_classes(args.cls_names)

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
        # 预处理代码
        img, im_size = preproc_ppyolo(origin_img, input_shape)

        session = onnxruntime.InferenceSession(args.model)
        # aaaaaaaaa = session.get_inputs()

        ort_inputs = {session.get_inputs()[0].name: img, session.get_inputs()[1].name: im_size}
        outputs = session.run(None, ort_inputs)
        output = outputs[0]
        yolo_boxes = output[:, :, :4]   # [N, A,  4]
        yolo_scores = output[:, :, 4:]  # [N, A, 80]

        nms_cfg = dict(
            score_threshold=0.01,
            post_threshold=0.01,
            nms_top_k=500,
            keep_top_k=100,
            use_gaussian=False,
            gaussian_sigma=2.,
        )

        # nms
        preds = []
        batch_size = yolo_boxes.shape[0]
        for i in range(batch_size):
            pred = numpy_matrix_nms(yolo_boxes[i, :, :], yolo_scores[i, :, :], **nms_cfg)
            preds.append(pred)
        dets = preds[0]
        if dets[0][0] > -0.5:
            final_boxes, final_scores, final_cls_inds = dets[:, 2:], dets[:, 1], dets[:, 0]
            origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                             conf=args.score_thr, class_names=class_names)
    else:
        raise NotImplementedError("Architectures \'{}\' is not implemented.".format(args.archi_name))

    mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, args.image_path.split("/")[-1])
    cv2.imwrite(output_path, origin_img)
