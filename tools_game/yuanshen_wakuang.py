#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2

import torch
import paddlehub as hub
import pyautogui

import numpy as np
import time
import autopy
import win32gui
import win32api
import win32con
# from tools_game.tools_yuanshen import *
import datetime as dt

import shutil
import cv2

from collections import deque
import datetime
import random
import cv2
import os
import time
import threading
import argparse
import textwrap


# add python path of this repo to sys.path
import sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from mmdet.data.data_augment import *
from mmdet.exp import get_exp
from mmdet.utils import fuse_model, get_model_info, postprocess, vis, get_classes, vis2

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("MieMieDetection Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

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
    return parser



# ?????????????????????????????????????????????
def find_flash_window(name):
    hwnd = win32gui.FindWindow(None, name)
    if (hwnd):
        win32gui.SetForegroundWindow(hwnd)
        rect = win32gui.GetWindowRect(hwnd)
        return rect
    return None


# ??????????????????
def mouse_click(x, y):
    x = int(x)
    y = int(y)
    win32api.SetCursorPos([x, y])
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    time.sleep(0.01)

# ??????????????????
def mouse_long_press(x, y, t):
    x = int(x)
    y = int(y)
    win32api.SetCursorPos([x, y])
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(t)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    time.sleep(0.01)


def get_key_value(key):
    v = 0
    if len(key) == 1:
        if ord("A") <= ord(key) and ord(key) <= ord("Z"):
            v = ord(key)

    if key == '->':
        v = 39
    elif key == '<-':
        v = 37
    elif key == 'up':
        v = 38
    elif key == 'down':
        v = 40
    # elif key == 'M':
    #     v = 77
    return v


# ??????????????????
def key_long_press(key, t=0):
    v = get_key_value(key)
    win32api.keybd_event(v, 0, 0, 0)  # ??????
    if t > 0:
        time.sleep(t)
    win32api.keybd_event(v, 0, win32con.KEYEVENTF_KEYUP, 0)  # ??????
    time.sleep(0.01)

# ??????????????????
def key_press(key):
    v = get_key_value(key)
    win32api.keybd_event(v, 0, 0, 0)  # ??????
    time.sleep(0.01)

# ??????????????????
def key_release(key):
    v = get_key_value(key)
    win32api.keybd_event(v, 0, win32con.KEYEVENTF_KEYUP, 0)  # ??????
    time.sleep(0.01)


# ????????????
def where_is_here_yuanshen(result, image):
    data = result['data']
    here = 'unknow'

    # ??????????????????????????????
    have_zjm1 = 0
    have_zjm2 = 0
    have_zjm3 = 0
    have_zjm4 = 0
    have_zjm5 = 0


    # ?????????????????????????????????
    have_map1 = 0
    have_map2 = 0
    have_map3 = 0
    have_map4 = 0
    have_map5 = 0
    have_map6 = 0
    have_map7 = 0


    for d in data:
        text = d['text']
        text_box_position = d['text_box_position']
        x0 = text_box_position[0][0]
        y0 = text_box_position[0][1]
        x1 = text_box_position[2][0]
        y1 = text_box_position[2][1]
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        if 'Enter' in text:
            if 26 < cx and cx < 108 and 703 < cy and cy < 731:
                have_zjm1 = 1
        if 'E' == text:
            if 1121 < cx and cx < 1145 and 709 < cy and cy < 727:
                have_zjm2 = 1
        if 'Q' == text:
            if 1204 < cx and cx < 1229 and 709 < cy and cy < 727:
                have_zjm3 = 1
        if 'ms' in text:
            if 1202 < cx and cx < 1259 and 80 < cy and cy < 95:
                have_zjm4 = 1
        if 'UID' in text:
            if 1126 < cx and cx < 1244 and 722 < cy and cy < 744:
                have_zjm5 = 1

        if '???????????????' in text:
            if 10 < cx and cx < 93 and 672 < cy and cy < 691:
                have_map1 = 1
        if '???' in text and '???' in text and '???' in text:
            if 1063 < cx and cx < 1184 and 42 < cy and cy < 70:
                have_map2 = 1
        if '??????????????????' in text:
            if 57 < cx and cx < 143 and 61 < cy and cy < 270:
                have_map3 = 1
        if '????????????' in text:
            if 57 < cx and cx < 143 and 61 < cy and cy < 270:
                have_map4 = 1
        if '????????????' in text:
            if 57 < cx and cx < 143 and 61 < cy and cy < 270:
                have_map5 = 1
        if '????????????' in text:
            if 57 < cx and cx < 143 and 61 < cy and cy < 270:
                have_map6 = 1
        if '????????????' in text:
            if 57 < cx and cx < 143 and 61 < cy and cy < 270:
                have_map7 = 1



    if (have_zjm1 + have_zjm2 + have_zjm3 + have_zjm4 + have_zjm5) > 2:
        here = 'zhujiemian'
    if (have_map1 + have_map2 + have_map3 + have_map4 + have_map5 + have_map6 + have_map7) > 3:
        here = 'map'

    if here == 'unknow':
        area = image[200:500, 20:320, :]
        std = np.std(area)
        if std < 10:
            here = 'loading'
    return here, data




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

        # ???????????????????????????
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
        # matrixNMS??????????????????-1?????????????????????
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

        # ???????????????????????????
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
        # matrixNMS??????????????????-1?????????????????????
        if output[0][0] < -0.5:
            return img
        output = output.cpu()

        bboxes = output[:, 2:6]

        cls = output[:, 0]
        scores = output[:, 1]

        # vis_res = vis2(img, bboxes, scores, cls, cls_conf, self.cls_names)
        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def detect(predictor, screen_shot):
    outputs, img_info = predictor.inference(screen_shot)
    output = outputs[0]
    ratio = img_info["ratio"]
    img = img_info["raw_img"]
    if output is None:
        return [], [], []
    output = output.cpu()
    bboxes = output[:, 0:4]

    # preprocessing: resize
    bboxes /= ratio

    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]
    bboxes = bboxes.cpu().detach().numpy()
    cls = cls.cpu().detach().numpy().astype(np.int32)
    cls_name = [predictor.cls_names[cid] for cid in cls]
    scores = scores.cpu().detach().numpy()
    return bboxes, cls_name, scores


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

    # ????????????
    archi_name = exp.archi_name

    # ???????????????????????????????????????????????????????????????elif
    if archi_name == 'YOLOX':
        if args.conf is not None:
            exp.test_conf = args.conf
        if args.nms is not None:
            exp.nmsthre = args.nms
        if args.tsize is not None:
            exp.test_size = (args.tsize, args.tsize)
    elif archi_name == 'PPYOLO':
        # PPYOLO????????????matrix_nms?????????matrix_nms????????????
        if args.conf is not None:
            exp.nms_cfg['score_threshold'] = args.conf
            exp.nms_cfg['post_threshold'] = args.conf
        if args.tsize is not None:
            exp.test_size = (args.tsize, args.tsize)
    elif archi_name == 'FCOS':
        # FCOS??????????????????matrix_nms?????????matrix_nms????????????
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
    # ???????????????????????????????????????????????????????????????elif
    if archi_name == 'YOLOX':
        # ??????????????????
        if not args.trt:
            if args.ckpt is None:
                ckpt_file = os.path.join(file_name, "best_ckpt.pth")
            else:
                ckpt_file = args.ckpt
            logger.info("loading checkpoint")
            ckpt = torch.load(ckpt_file, map_location="cpu")
            # load the model state dict
            model.load_state_dict(ckpt["model"])
            logger.info("loaded checkpoint done.")

        # ????????????bn???????????????????????????
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

        isDebug = True if sys.gettrace() else False
        if isDebug:
            print('Debug Mode.')
        else:
            exp.cls_names = '../' + exp.cls_names
        predictor = YOLOXPredictor(
            model, exp, trt_file, decoder,
            args.device, args.fp16, args.legacy,
        )
    elif archi_name == 'PPYOLO':
        # ??????????????????
        if not args.trt:
            if args.ckpt is None:
                ckpt_file = os.path.join(file_name, "best_ckpt.pth")
            else:
                ckpt_file = args.ckpt
            logger.info("loading checkpoint")
            ckpt = torch.load(ckpt_file, map_location="cpu")
            # load the model state dict
            model.load_state_dict(ckpt["model"])
            logger.info("loaded checkpoint done.")

        # ????????????bn???????????????????????????
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
    elif archi_name == 'FCOS':
        # ??????????????????
        if not args.trt:
            if args.ckpt is None:
                ckpt_file = os.path.join(file_name, "best_ckpt.pth")
            else:
                ckpt_file = args.ckpt
            logger.info("loading checkpoint")
            ckpt = torch.load(ckpt_file, map_location="cpu")
            # load the model state dict
            model.load_state_dict(ckpt["model"])
            logger.info("loaded checkpoint done.")

        # ????????????bn???????????????????????????
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
    logger.info('Start...............')
    use_gpu = args.device == "gpu"
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # ??????gpu
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # ????????????cpu????????????????????????

    # ??????????????????????????????
    ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")
    # ?????????????????????????????????????????????
    # ocr = hub.Module(name="chinese_ocr_fdb_crnn_server")

    output_dir = 'ocr_result'

    # ????????????????????????1280x720
    print("finding...")
    pos = find_flash_window("??????")
    if (pos == None):
        print("unfound!")
        exit()
    print("get!")
    xbase = pos[0]
    ybase = pos[1]
    xend = pos[2]
    yend = pos[3]
    game_w = xend - xbase  # ????????????
    game_h = yend - ybase  # ????????????
    game_half_w = game_w // 2  # ?????????????????????
    game_half_h = game_h // 2  # ?????????????????????

    dir_name = 'yuanshen_temp'
    if os.path.exists('D://%s/' % (dir_name,)):
        shutil.rmtree('D://%s/' % (dir_name,))
    os.mkdir('D://%s/' % (dir_name,))

    if os.path.exists(output_dir): shutil.rmtree(output_dir)

    from pynput import keyboard
    break_program = False

    # ??????End??????????????????
    def on_press(key):
        global break_program
        print(key)
        if key == keyboard.Key.end:
            print('end pressed')
            break_program = True
            return False

    target_positions = []  # ?????????
    tp_offset = []  # ??????????????????????????????
    tag_obj_num = []  # ?????????????????????????????????????????????
    tag_obj_path = []  # ??????????????????????????????????????????+??????
    tag_obj_area = []  # ?????????????????????????????????????????????????????????????????????????????????

    # ================= ?????????????????????????????? =================

    # ???????????????????????????ocr?????????????????????????????????????????????
    target_positions.append(['????????????', '????????????', '????????????'])
    tp_offset.append([[-10, 35], [-10, 35], [52, -111]])
    tag_obj_num.append(2)
    tag_obj_path.append([[['W', 7.5], ['D', 0.8]], None])
    tag_obj_area.append([None, 149941])


    # ================= ??????????????????????????????????????? =================

    tpi = 0  # ????????????????????????
    toi = 0  # ???????????????????????????
    face_tag_obj = False  # ??????????????????????????????
    target_positions_len = len(target_positions)

    # ????????????????????????????????????????????????
    reach_nearest_anchor = False
    # ???????????????????????????
    reach_mine = False
    # ??????????????????????????????
    mine_all = False
    # ???????????????????????????????????????
    max_mine_time = 20

    with keyboard.Listener(on_press=on_press) as listener:
        # ???????????????????????????
        jiaoben_start_time = time.time()
        while break_program == False:
            img = pyautogui.screenshot(region=[xbase, ybase, xend - xbase, yend - ybase])  # x,y,w,h
            screen_shot = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            image_name = 'D://%s/temp2.png' % (dir_name,)
            cv2.imwrite(image_name, screen_shot)

            # ???????????????
            np_images = [screen_shot]
            results = ocr.recognize_text(
                images=np_images,  # ???????????????ndarray.shape ??? [H, W, C]???BGR?????????
                use_gpu=use_gpu,  # ???????????? GPU????????????GPU???????????????CUDA_VISIBLE_DEVICES????????????
                output_dir=output_dir,  # ???????????????????????????????????? ocr_result???
                visualization=True,  # ?????????????????????????????????????????????
                box_thresh=0.5,  # ????????????????????????????????????
                text_thresh=0.5)  # ???????????????????????????????????????
            here, data = where_is_here_yuanshen(results[0], screen_shot)
            print(here)
            print(data)

            # ????????????????????????????????????
            # reach_nearest_anchor = True
            # reach_mine = True
            # mine_start_time = time.time()
            # ????????????????????????????????????????????????????????????
            if not reach_mine and not reach_nearest_anchor:
                if here == 'zhujiemian':
                    key_long_press('M', t=0.2)
                    time.sleep(2.0)
                elif here == 'map':
                    for d in data:
                        text = d['text']
                        text_box_position = d['text_box_position']
                        x0 = text_box_position[0][0]
                        y0 = text_box_position[0][1]
                        x1 = text_box_position[2][0]
                        y1 = text_box_position[2][1]
                        cx = (x0 + x1) / 2
                        cy = (y0 + y1) / 2
                        if text in target_positions[tpi]:
                            found = False
                            for k, tp in enumerate(target_positions[tpi]):
                                if tp != text:
                                    continue
                                xxx = cx + tp_offset[tpi][k][0]
                                yyy = cy + tp_offset[tpi][k][1]
                                if 10 < xxx and xxx < game_w-10 and 30 < yyy and yyy < game_h-30:
                                    found = True
                                    mouse_click(xbase + xxx, ybase + yyy)
                                    time.sleep(1.0)
                                    mouse_click(xbase + 1130, ybase + 700)  # ????????????
                                    time.sleep(1.0)
                                    reach_nearest_anchor = True
                                    toi = 0
                                    face_tag_obj = False
                            if found:
                                break
                elif here == 'loading':
                    time.sleep(1.0)
            # ???????????????????????????????????????????????????????????????
            elif not reach_mine and reach_nearest_anchor:
                if here == 'zhujiemian':
                    # ???????????????????????????????????????????????????
                    tag_area = 0.0
                    while not face_tag_obj:
                        img = pyautogui.screenshot(region=[xbase, ybase, xend - xbase, yend - ybase])  # x,y,w,h
                        screen_shot = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                        bboxes, cls_name, scores = detect(predictor, screen_shot)
                        # ?????????????????????????????????
                        if 'tag_obj' not in cls_name:
                            nx, ny = 500, 0
                            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, nx, ny)
                        # ?????????????????????????????????????????????????????????????????????
                        else:
                            bbox_i = 0
                            for r, ccc in enumerate(cls_name):
                                if ccc == 'tag_obj':
                                    bbox_i = r
                                    break
                            bbox = bboxes[bbox_i]
                            tag_obj_cx = (bbox[0] + bbox[2]) / 2.0
                            tag_obj_cy = (bbox[1] + bbox[3]) / 2.0
                            tag_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                            chajv_x = tag_obj_cx - game_half_w
                            chajv_y = game_half_h - tag_obj_cy
                            print('chajv_x:%f' % chajv_x)

                            if abs(chajv_x) < 50:
                                print('?????????????????????????????????')
                                face_tag_obj = True
                            elif chajv_x > 0:
                                nx, ny = 50, 0
                                if abs(chajv_x) < 100:
                                    nx, ny = 20, 0
                                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, nx, ny)
                            elif chajv_x < 0:
                                nx, ny = -50, 0
                                if abs(chajv_x) < 100:
                                    nx, ny = -20, 0
                                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, nx, ny)
                    # tag_obj_path????????????None?????????????????????
                    if tag_obj_path[tpi][toi] is not None:
                        paths = tag_obj_path[tpi][toi]
                        for _path in paths:
                            key_long_press(_path[0], t=_path[1])
                        # ?????????????????????????????????????????????????????????????????????
                        toi += 1
                        # ?????????????????????????????????
                        face_tag_obj = False
                    # tag_obj_path?????????None????????????????????????????????????????????????
                    else:
                        # ??????????????????????????????????????????????????????????????????
                        if tag_area > tag_obj_area[tpi][toi]:
                            toi += 1
                            face_tag_obj = False
                        else:  # ??????????????????????????????????????????
                            key_long_press('W', t=2)
                            face_tag_obj = False  # ????????????????????????????????????
                    if toi >= tag_obj_num[tpi]:
                        print('?????????????????????!')
                        reach_mine = True
                        mine_all = False
                        mine_start_time = time.time()
            # ???????????????????????????
            elif reach_mine:
                # ??????????????????????????????
                while not mine_all:
                    img = pyautogui.screenshot(region=[xbase, ybase, xend - xbase, yend - ybase])  # x,y,w,h
                    screen_shot = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                    bboxes, cls_name, scores = detect(predictor, screen_shot)
                    # ???????????????????????????????????????
                    if 'kuangshi' not in cls_name:
                        nx, ny = 500, 0
                        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, nx, ny)

                        rrrrrr = random.choice(range(4))
                        if rrrrrr == 0:
                            key_long_press('W', t=2)
                        if rrrrrr == 1:
                            key_long_press('A', t=2)
                        if rrrrrr == 2:
                            key_long_press('S', t=2)
                        if rrrrrr == 3:
                            key_long_press('D', t=2)
                    # ????????????????????????????????????????????????????????????
                    else:
                        bbox_i = 0
                        for r, ccc in enumerate(cls_name):
                            if ccc == 'kuangshi':
                                bbox_i = r
                                break
                        bbox = bboxes[bbox_i]
                        kuangshi_cx = (bbox[0] + bbox[2]) / 2.0
                        kuangshi_cy = (bbox[1] + bbox[3]) / 2.0
                        kuangshi_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        chajv_x = kuangshi_cx - game_half_w
                        chajv_y = game_half_h - kuangshi_cy
                        print('chajv_x:%f' % chajv_x)

                        if abs(chajv_x) < 50:
                            pass
                        elif chajv_x > 0:
                            nx, ny = 50, 0
                            if abs(chajv_x) < 100:
                                nx, ny = 20, 0
                            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, nx, ny)
                        elif chajv_x < 0:
                            nx, ny = -50, 0
                            if abs(chajv_x) < 100:
                                nx, ny = -20, 0
                            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, nx, ny)
                        key_long_press('W', t=1)
                        mouse_click(xbase + 1100, ybase + 680)
                        time.sleep(0.5)
                        for _ in range(3):
                            key_long_press('F', t=0.2)
                            time.sleep(0.5)
                    cost_time = time.time() - mine_start_time
                    print('cost time: {0:.6f}s'.format(cost_time))
                    if cost_time > max_mine_time:
                        mine_all = True
                        tpi += 1
                        if tpi >= len(target_positions):
                            print('???????????????????????????!')
                            time.sleep(7)
                            break_program = True

        # =============================== Done ===============================
        print('Done.')
        listener.join()




if __name__ == "__main__":
    args = make_parser().parse_args()
    # ???????????????????????????
    isDebug = True if sys.gettrace() else False
    # if isDebug:
    #     print('Debug Mode.')
    #     args.exp_file = '../' + args.exp_file
    #     args.ckpt = '../' + args.ckpt   # ?????????????????????????????????????????????
    #     args.path = '../' + args.path   # ?????????????????????????????????????????????
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
