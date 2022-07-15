#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import copy
import os
import time
from loguru import logger

import cv2

import torch

# add python path of this repo to sys.path
import sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from mmdet.data.data_augment import *
from mmdet.exp import get_exp
from mmdet.utils import fuse_model, get_model_info, postprocess, vis, get_classes, vis2, vis_solo, load_ckpt
import mmdet.models.ncnn_utils as ncnn_utils

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("MieMieDetection Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, ncnn, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--ncnn_output_path", default="", help="path to save ncnn model."
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class YOLOXPredictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = get_classes(exp.cls_names)
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        output = output[0]
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


class PPYOLOPredictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = get_classes(exp.cls_names)
        self.num_classes = exp.num_classes
        self.confthre = exp.nms_cfg['post_threshold']
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16

        # 预测时的数据预处理
        self.context = exp.context
        self.to_rgb = exp.decodeImage['to_rgb']
        target_size = self.test_size[0]
        resizeImage = ResizeImage(target_size=target_size, interp=exp.resizeImage['interp'])
        normalizeImage = NormalizeImage(**exp.normalizeImage)
        permute = Permute(**exp.permute)
        self.preproc = PPYOLOValTransform(self.context, self.to_rgb, resizeImage, normalizeImage, permute)

        # TensorRT
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, im_size = self.preproc(img)
        img = torch.from_numpy(img)
        im_size = torch.from_numpy(im_size)
        img = img.float()
        im_size = im_size.float()
        if self.device == "gpu":
            img = img.cuda()
            im_size = im_size.cuda()
            if self.fp16:
                img = img.half()  # to FP16
                im_size = im_size.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img, im_size)
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        output = output[0]
        img = img_info["raw_img"]
        # matrixNMS返回的结果为-1时表示没有物体
        if output[0][0] < -0.5:
            return img
        output = output.cpu()

        bboxes = output[:, 2:6]

        cls = output[:, 0]
        scores = output[:, 1]

        # vis_res = vis2(img, bboxes, scores, cls, cls_conf, self.cls_names)
        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


class SOLOPredictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = get_classes(exp.cls_names)
        self.num_classes = exp.num_classes
        self.confthre = exp.nms_cfg['post_threshold']
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16

        # 预测时的数据预处理
        self.context = exp.context
        self.to_rgb = exp.decodeImage['to_rgb']
        target_size = self.test_size[0]
        resizeImage_cfg = copy.deepcopy(exp.resizeImage)
        resizeImage_cfg['target_size'] = target_size
        resizeImage = ResizeImage(**resizeImage_cfg)
        normalizeImage = NormalizeImage(**exp.normalizeImage)
        permute = Permute(**exp.permute)
        padBatch = PadBatch(**exp.padBatch)
        self.preproc = SOLOValTransform(self.context, self.to_rgb, resizeImage, normalizeImage, permute, padBatch)

        # TensorRT
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, im_size, ori_shape = self.preproc(img)
        img = torch.from_numpy(img)
        im_size = torch.from_numpy(im_size)
        ori_shape = torch.from_numpy(ori_shape)
        img = img.float()
        im_size = im_size.float()
        ori_shape = ori_shape.float()
        if self.device == "gpu":
            img = img.cuda()
            im_size = im_size.cuda()
            ori_shape = ori_shape.cuda()
            if self.fp16:
                img = img.half()  # to FP16
                im_size = im_size.half()  # to FP16
                ori_shape = ori_shape.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img, im_size, ori_shape)
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        bbox_num = output['bbox_num'][0]
        img = img_info["raw_img"]
        if bbox_num > 0:
            masks = output['segm'][0]
            cls = output['cate_label'][0]
            scores = output['cate_score'][0]

            masks = masks.cpu().detach().numpy()
            cls = cls.to(torch.int32).cpu().detach().numpy()
            scores = scores.cpu().detach().numpy()

            # 获取boxes
            boxes = []
            for ms in masks:
                sum_1 = np.sum(ms, axis=0)
                x = np.where(sum_1 > 0.5)[0]
                sum_2 = np.sum(ms, axis=1)
                y = np.where(sum_2 > 0.5)[0]
                if len(x) == 0:  # 掩码全是0的话（即没有一个像素是前景）
                    x0, x1, y0, y1 = 0, 1, 0, 1
                else:
                    x0, x1, y0, y1 = x[0], x[-1], y[0], y[-1]
                boxes.append([x0, y0, x1, y1])
            bboxes = np.array(boxes).astype(np.float32)
            vis_res = vis_solo(img, bboxes, masks, scores, cls, cls_conf, self.cls_names)
            return vis_res
        else:
            return img


class PPYOLOEPredictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = get_classes(exp.cls_names)
        self.num_classes = exp.num_classes
        self.confthre = exp.nms_cfg['score_threshold']
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16

        # 预测时的数据预处理
        self.context = exp.context
        self.to_rgb = exp.decodeImage['to_rgb']
        target_size = self.test_size[0]
        resizeImage = ResizeImage(target_size=target_size, interp=exp.resizeImage['interp'])
        normalizeImage = NormalizeImage(**exp.normalizeImage)
        permute = Permute(**exp.permute)
        self.preproc = PPYOLOEValTransform(self.context, self.to_rgb, resizeImage, normalizeImage, permute)

        # TensorRT
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, scale_factor = self.preproc(img)
        img = torch.from_numpy(img)
        scale_factor = torch.from_numpy(scale_factor)
        img = img.float()
        scale_factor = scale_factor.float()
        if self.device == "gpu":
            img = img.cuda()
            scale_factor = scale_factor.cuda()
            if self.fp16:
                img = img.half()  # to FP16
                scale_factor = scale_factor.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img, scale_factor)
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        output = output[0]
        img = img_info["raw_img"]
        # matrixNMS返回的结果为-1时表示没有物体
        if output[0][0] < -0.5:
            return img
        output = output.cpu()

        bboxes = output[:, 2:6]

        cls = output[:, 0]
        scores = output[:, 1]

        # vis_res = vis2(img, bboxes, scores, cls, cls_conf, self.cls_names)
        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


class FCOSPredictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = get_classes(exp.cls_names)
        self.num_classes = exp.num_classes
        self.confthre = exp.nms_cfg['post_threshold']
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16

        # 预测时的数据预处理
        self.context = exp.context

        # sample_transforms
        self.to_rgb = exp.decodeImage['to_rgb']
        normalizeImage = NormalizeImage(**exp.normalizeImage)
        target_size = self.test_size[0]
        max_size = self.test_size[1]
        resizeImage = ResizeImage(target_size=target_size, resize_box=False, interp=exp.resizeImage['interp'],
                                  max_size=max_size, use_cv2=exp.resizeImage['use_cv2'])
        permute = Permute(**exp.permute)

        # batch_transforms
        padBatch = PadBatch(**exp.padBatch)

        self.preproc = FCOSValTransform(self.context, self.to_rgb, normalizeImage, resizeImage, permute, padBatch)

        # TensorRT
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, im_scale = self.preproc(img)
        img = torch.from_numpy(img)
        im_scale = torch.from_numpy(im_scale)
        img = img.float()
        im_scale = im_scale.float()
        if self.device == "gpu":
            img = img.cuda()
            im_scale = im_scale.cuda()
            if self.fp16:
                img = img.half()  # to FP16
                im_scale = im_scale.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img, im_scale)
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        output = output[0]
        img = img_info["raw_img"]
        # matrixNMS返回的结果为-1时表示没有物体
        if output[0][0] < -0.5:
            return img
        output = output.cpu()

        bboxes = output[:, 2:6]

        cls = output[:, 0]
        scores = output[:, 1]

        # vis_res = vis2(img, bboxes, scores, cls, cls_conf, self.cls_names)
        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result_image = predictor.visual(outputs, img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    # 算法名字
    archi_name = exp.archi_name

    # 不同的算法输入不同，新增算法时这里也要增加elif
    if archi_name == 'YOLOX':
        if args.conf is not None:
            exp.test_conf = args.conf
        if args.nms is not None:
            exp.nmsthre = args.nms
        if args.tsize is not None:
            exp.test_size = (args.tsize, args.tsize)
    elif archi_name == 'PPYOLO':
        # PPYOLO使用的是matrix_nms，修改matrix_nms的配置。
        if args.conf is not None:
            exp.nms_cfg['score_threshold'] = args.conf
            exp.nms_cfg['post_threshold'] = args.conf
        if args.tsize is not None:
            exp.test_size = (args.tsize, args.tsize)
    elif archi_name == 'PPYOLOE':
        # PPYOLOE使用的是multiclass_nms，修改multiclass_nms的配置。
        if args.conf is not None:
            exp.nms_cfg['score_threshold'] = args.conf
        if args.tsize is not None:
            exp.test_size = [args.tsize, args.tsize]
            exp.head['eval_size'] = exp.test_size
    elif archi_name == 'SOLO':
        # SOLO使用的是matrix_nms，修改matrix_nms的配置。
        if args.conf is not None:
            exp.nms_cfg['score_threshold'] = args.conf
            exp.nms_cfg['post_threshold'] = args.conf
        if args.tsize is not None:
            exp.test_size = (args.tsize, args.tsize)
    elif archi_name == 'FCOS':
        # FCOS暂时使用的是matrix_nms，修改matrix_nms的配置。
        if args.conf is not None:
            exp.nms_cfg['score_threshold'] = args.conf
            exp.nms_cfg['post_threshold'] = args.conf
        if args.tsize is not None:
            exp.test_size = (args.tsize, exp.test_size[1])
    else:
        raise NotImplementedError("Architectures \'{}\' is not implemented.".format(archi_name))

    model = exp.get_model()
    # logger.info("Model Summary: {}".format(get_model_info(archi_name, model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    predictor = None
    # 不同的算法输入不同，新增算法时这里也要增加elif
    if archi_name == 'YOLOX':
        # 加载模型权重
        if not args.trt:
            if args.ckpt is None:
                ckpt_file = os.path.join(file_name, "best_ckpt.pth")
            else:
                ckpt_file = args.ckpt
            logger.info("loading checkpoint")
            ckpt = torch.load(ckpt_file, map_location="cpu")
            # load the model state dict
            model = load_ckpt(model, ckpt["model"])
            logger.info("loaded checkpoint done.")

        # 卷积层和bn层合并为一个卷积层
        if args.fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)

        if args.trt:
            assert not args.fuse, "TensorRT model is not support model fusing!"
            trt_file = os.path.join(file_name, "model_trt.pth")
            assert os.path.exists(
                trt_file
            ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
            model.head.decode_in_inference = False
            decoder = model.head.decode_outputs
            logger.info("Using TensorRT to inference")
        else:
            trt_file = None
            decoder = None

        predictor = YOLOXPredictor(
            model, exp, trt_file, decoder,
            args.device, args.fp16, args.legacy,
        )
    elif archi_name == 'PPYOLO':
        # 加载模型权重
        if not args.trt:
            if args.ckpt is None:
                ckpt_file = os.path.join(file_name, "best_ckpt.pth")
            else:
                ckpt_file = args.ckpt
            logger.info("loading checkpoint")
            ckpt = torch.load(ckpt_file, map_location="cpu")
            # load the model state dict
            model = load_ckpt(model, ckpt["model"])
            logger.info("loaded checkpoint done.")

        # 卷积层和bn层合并为一个卷积层
        if args.fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)

        if args.trt:
            assert not args.fuse, "TensorRT model is not support model fusing!"
            trt_file = os.path.join(file_name, "model_trt.pth")
            assert os.path.exists(
                trt_file
            ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
            model.head.decode_in_inference = False
            logger.info("Using TensorRT to inference")
        else:
            trt_file = None

        predictor = PPYOLOPredictor(
            model, exp, trt_file,
            args.device, args.fp16, args.legacy,
        )
    elif archi_name == 'PPYOLOE':
        # 加载模型权重
        if not args.trt:
            if args.ckpt is None:
                ckpt_file = os.path.join(file_name, "best_ckpt.pth")
            else:
                ckpt_file = args.ckpt
            logger.info("loading checkpoint")
            ckpt = torch.load(ckpt_file, map_location="cpu")
            # load the model state dict
            model = load_ckpt(model, ckpt["model"])
            logger.info("loaded checkpoint done.")

        # 卷积层和bn层合并为一个卷积层
        if args.fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)

        if args.trt:
            assert not args.fuse, "TensorRT model is not support model fusing!"
            trt_file = os.path.join(file_name, "model_trt.pth")
            assert os.path.exists(
                trt_file
            ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
            model.head.decode_in_inference = False
            logger.info("Using TensorRT to inference")
        else:
            trt_file = None

        predictor = PPYOLOEPredictor(
            model, exp, trt_file,
            args.device, args.fp16, args.legacy,
        )
    elif archi_name == 'SOLO':
        # 加载模型权重
        if not args.trt:
            if args.ckpt is None:
                ckpt_file = os.path.join(file_name, "best_ckpt.pth")
            else:
                ckpt_file = args.ckpt
            logger.info("loading checkpoint")
            ckpt = torch.load(ckpt_file, map_location="cpu")
            # load the model state dict
            model = load_ckpt(model, ckpt["model"])
            logger.info("loaded checkpoint done.")

        # 卷积层和bn层合并为一个卷积层
        if args.fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)

        if args.trt:
            assert not args.fuse, "TensorRT model is not support model fusing!"
            trt_file = os.path.join(file_name, "model_trt.pth")
            assert os.path.exists(
                trt_file
            ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
            model.head.decode_in_inference = False
            logger.info("Using TensorRT to inference")
        else:
            trt_file = None

        predictor = SOLOPredictor(
            model, exp, trt_file,
            args.device, args.fp16, args.legacy,
        )
    elif archi_name == 'FCOS':
        # 加载模型权重
        if not args.trt:
            if args.ckpt is None:
                ckpt_file = os.path.join(file_name, "best_ckpt.pth")
            else:
                ckpt_file = args.ckpt
            logger.info("loading checkpoint")
            ckpt = torch.load(ckpt_file, map_location="cpu")
            # load the model state dict
            model = load_ckpt(model, ckpt["model"])
            logger.info("loaded checkpoint done.")

        # 卷积层和bn层合并为一个卷积层
        if args.fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)

        if args.trt:
            assert not args.fuse, "TensorRT model is not support model fusing!"
            trt_file = os.path.join(file_name, "model_trt.pth")
            assert os.path.exists(
                trt_file
            ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
            model.head.decode_in_inference = False
            logger.info("Using TensorRT to inference")
        else:
            trt_file = None

        predictor = FCOSPredictor(
            model, exp, trt_file,
            args.device, args.fp16, args.legacy,
        )
    else:
        raise NotImplementedError("Architectures \'{}\' is not implemented.".format(archi_name))
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)
    elif args.demo == "ncnn":
        bp = open('%s.bin' % args.ncnn_output_path, 'wb')
        pp = ''
        layer_id = 0
        tensor_id = 0
        tensor_names = []
        if archi_name == 'YOLOX':
            raise NotImplementedError("Architectures \'{}\' is not implemented.".format(archi_name))
        elif archi_name == 'PPYOLO':
            raise NotImplementedError("Architectures \'{}\' is not implemented.".format(archi_name))
        elif archi_name == 'PPYOLOE':
            pp += 'Input\tlayer_%.8d\t0 1 tensor_%.8d\n' % (layer_id, tensor_id)
            layer_id += 1
            tensor_id += 1
        else:
            raise NotImplementedError("Architectures \'{}\' is not implemented.".format(archi_name))
        ncnn_data = {}
        ncnn_data['bp'] = bp
        ncnn_data['pp'] = pp
        ncnn_data['layer_id'] = layer_id
        ncnn_data['tensor_id'] = tensor_id
        bottom_names = ncnn_utils.newest_bottom_names(ncnn_data)
        bottom_names = model.export_ncnn(ncnn_data, bottom_names)
        # 如果1个张量作为了n(n>1)个层的输入张量，应该用Split层将它复制n份，每1层用掉1个。
        bottom_names = ncnn_utils.split_input_tensor(ncnn_data, bottom_names)
        pp = ncnn_data['pp']
        layer_id = ncnn_data['layer_id']
        tensor_id = ncnn_data['tensor_id']
        pp = pp.replace('tensor_%.8d' % (0,), 'images')
        pp = pp.replace(bottom_names[0], 'cls_score')
        pp = pp.replace(bottom_names[1], 'reg_dist')
        pp = '7767517\n%d %d\n'%(layer_id, tensor_id) + pp
        with open('%s.param' % args.ncnn_output_path, 'w', encoding='utf-8') as f:
            f.write(pp)
            f.close()
        logger.info("Saving ncnn param file in %s.param" % args.ncnn_output_path)
        logger.info("Saving ncnn bin file in %s.bin" % args.ncnn_output_path)


if __name__ == "__main__":
    args = make_parser().parse_args()
    # 判断是否是调试状态
    isDebug = True if sys.gettrace() else False
    if isDebug:
        print('Debug Mode.')
        args.exp_file = '../' + args.exp_file
        args.ckpt = '../' + args.ckpt   # 如果是绝对路径，把这一行注释掉
        args.path = '../' + args.path   # 如果是绝对路径，把这一行注释掉
        args.ncnn_output_path = '../' + args.ncnn_output_path   # 如果是绝对路径，把这一行注释掉
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
