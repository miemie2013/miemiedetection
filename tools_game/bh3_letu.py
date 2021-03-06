#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2
import easyocr
import pyautogui
import win32gui
import win32api
import win32con
import datetime as dt

import torch

# add python path of this repo to sys.path
import sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from mmdet.data.data_augment import *
from mmdet.exp import get_exp
from mmdet.utils import fuse_model, get_model_info, postprocess, vis, get_classes, vis2, load_ckpt, setup_logger

from tools_game.tools import find_flash_window, getText, where_is_here_letu, mouse_click, mouse_long_press, key_long_press


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("MieMieDetection Demo!")
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

    # ?????????0-?????????1-?????????2-?????????3-????????????4-??????
    parser.add_argument("--chara", default=0, type=int, help="chara")
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

        # ???????????????????????????
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


def image_demo(predictor, vis_folder, path, current_time, save_result, args):
    easyocr_reader = easyocr.Reader(['ch_sim', 'en'])  # this needs to run only once to load the model into memory

    # ????????????????????????1280x720
    logger.info("finding...")
    pos = find_flash_window("??????3")
    if (pos == None):
        logger.info("unfound!")
        exit()
    logger.info("get!")
    xbase = pos[0]
    ybase = pos[1]
    xend = pos[2]
    yend = pos[3]
    w = xend - xbase  # ????????????
    h = yend - ybase  # ????????????

    img_rgb = pyautogui.screenshot(region=[xbase, ybase, w, h])
    img_rgb = np.asarray(img_rgb)
    img_bgr = img_rgb[:, :, [2, 1, 0]]
    # cv2.imwrite('aaaaaaaa.jpg', img_bgr)

    break_program = False

    chara = args.chara

    # ?????????0-?????????1-?????????2-?????????3-?????????4-??????110???5-??????130
    nandu = 0

    use_gpu = False
    use_gpu = True

    # ????????????????????????
    shuamanjiuting = True
    shuamanjiuting = False

    duiyou_xy = [[700, 200], [860, 200], [1000, 200], [1180, 200],
                 [700, 330], [860, 330], [1000, 330], [1180, 330]]
    duiyou_ji = ['??????', '??????', '???', '??????', '???', '??????', '???', '???']

    # ???????????????????????????
    juese_name = ''
    # ??????????????????????????????????????????????????????
    ziji_coord = []
    # ?????????????????????duiyou_xy???????????????
    duiyou_1 = 0
    duiyou_2 = 0
    # ?????????????????????
    manyis = []
    shanghaileixing = ''
    if chara == 0:
        juese_name = '??????'
        shanghaileixing = '??????'
        ziji_coord = [430, 460]
        duiyou_1 = 0
        duiyou_2 = 1
        manyis.append('??????????????????????????????40%')
        manyis.append('?????????????????????????????????????????????')
        manyis.append('???????????????????????????')
        manyis.append('????????????')
        manyis.append('??????')
        manyis.append('????????????')
        manyis.append('?????????')
        manyis.append('??????')
        manyis.append('??????')
    elif chara == 1:
        # juese_name = '??????'
        juese_name = '???'
        shanghaileixing = '??????'
        # ziji_coord = [900, 320]
        ziji_coord = [650, 400]
        duiyou_1 = 0
        duiyou_2 = 3
        # manyis.append('???????????????????????????')
        manyis.append('??????')
        manyis.append('????????????')
        manyis.append('???????????????')
        manyis.append('???????????????????????????')
        manyis.append('????????????')
        manyis.append('??????')
        manyis.append('?????????')
        manyis.append('????????????')
        manyis.append('??????')
        manyis.append('??????')
    elif chara == 2:
        juese_name = '??????'
        shanghaileixing = '??????'
        ziji_coord = [410, 320]
        duiyou_1 = 6
        duiyou_2 = 3
        manyis.append('????????????')
        manyis.append('????????????????????????')
        manyis.append('???????????????????????????')
        manyis.append('????????????')
        manyis.append('??????')
        manyis.append('?????????')
        manyis.append('????????????')
        manyis.append('??????')
        manyis.append('??????')
    elif chara == 3:
        juese_name = '???'
        shanghaileixing = '??????'
        ziji_coord = [1150, 460]
        duiyou_1 = 3
        duiyou_2 = 6
        manyis.append('??????????????????')
        manyis.append('??????')
        manyis.append('???????????????????????????')
        manyis.append('????????????')
        manyis.append('??????')
        manyis.append('?????????')
        manyis.append('????????????')
        manyis.append('??????')
        manyis.append('??????')
    elif chara == 4:
        juese_name = '??????'
        shanghaileixing = '??????'
        ziji_coord = [180, 390]
        duiyou_1 = 0
        duiyou_2 = 1
        manyis.append('?????????????????????')
        manyis.append('???????????????????????????')
        manyis.append('???????????????????????????')
        manyis.append('????????????')
        manyis.append('??????')
        manyis.append('????????????')
        manyis.append('?????????')
        manyis.append('??????')
        manyis.append('??????')

    class_cn_names = []
    class_cn_names.append('????????????')  # id = 0
    class_cn_names.append('????????????')  # id = 1
    class_cn_names.append('????????????')  # id = 2
    class_cn_names.append('boss??????')
    class_cn_names.append('????????????')
    class_cn_names.append('?????????????????????')

    class_cn_names.append('??????')  # id = 6
    class_cn_names.append('??????')  # id = 7
    class_cn_names.append('?????????')
    class_cn_names.append('??????')
    class_cn_names.append('??????')
    class_cn_names.append('???')
    class_cn_names.append('??????')

    # ?????????????????????????????????????????????????????????
    maodian_cid = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


    attack_count = 0  # ????????????

    xuli = 10  # ????????????????????????
    bisha = 7  # ????????????????????????
    shangbijiange = 15  # ????????????

    # ????????????????????????
    dazhao_last_time = time.time() - 99
    # ???????????????????????????
    baofa_ing = False

    if chara == 0:
        xuli = 10  # ????????????????????????
        bisha = 7  # ????????????????????????
        shangbijiange = 15  # ????????????
    elif chara == 1:
        xuli = 10  # ????????????????????????
        bisha = 8  # ????????????????????????
        shangbijiange = 5  # ????????????
    elif chara == 2:
        xuli = 20  # ????????????????????????
        bisha = 12  # ????????????????????????
        shangbijiange = 5  # ????????????
    elif chara == 3:
        xuli = 20  # ????????????????????????
        bisha = 10  # ????????????????????????
        shangbijiange = 5  # ????????????
    elif chara == 4:
        xuli = 10  # ????????????????????????
        bisha = 7  # ????????????????????????
        shangbijiange = 15  # ????????????




    # ???????????????????????????
    jiaoben_start_time = time.time()
    while break_program == False:
        results = getText(easyocr_reader, xbase, ybase, w, h)
        here, data = where_is_here_letu(results)

        t_1 = 0.5
        if here == 'zhujiemian':
            logger.info('zhujiemian')
            mouse_click(xbase + 1111, ybase + 200)  # ??????
            time.sleep(1.)
            mouse_click(xbase + 70, ybase + 420)  # ??????
            time.sleep(1.)
            mouse_click(xbase + 1100, ybase + 400)  # ????????????
            time.sleep(5.)
        elif here == 'letuzhujiemian':
            logger.info('letuzhujiemian')
            mouse_click(xbase + 1140, ybase + 660)  # ????????????
            time.sleep(1.)
            mouse_click(xbase + 1000, ybase + 500)  # ????????????
            time.sleep(1.)

            # ?????????
            mouse_click(xbase + 875, ybase + 600)  #
            time.sleep(1.)
            key_long_press('up', t=1.5)
            mouse_click(xbase + 875, ybase + 370)  # ??????*1.5
            time.sleep(1.)
            mouse_click(xbase + 1111, ybase + 600)  # ????????????
            time.sleep(1.)


            keyin_id = 0
            keyin_list = []
            time.sleep(t_1)
            time.sleep(1.3)
        elif here == 'jixushangci':
            logger.info('jixushangci')
            mouse_click(xbase + 710, ybase + 552)  # ????????????
            time.sleep(t_1)
            mouse_click(xbase + 780, ybase + 450)  # ??????
            time.sleep(t_1)
            mouse_click(xbase + 670, ybase + 50)  # ??????
            time.sleep(t_1)
            mouse_click(xbase + 1140, ybase + 700)  # ????????????
            time.sleep(t_1)
            # mouse_click(xbase + 990, ybase + 552)  # ????????????
            # time.sleep(t_1)
            # keyin_id = 22
        elif here == 'zhandouzhunbei':
            logger.info('zhandouzhunbei')
            # ????????????????????????????????????
            cur_huobi_str = ''
            duizhangji1 = ''
            duizhangji2 = ''
            for d in data:
                text = d[0]
                text_box_position = d[1]
                x0 = text_box_position[0][0]
                y0 = text_box_position[0][1]
                x1 = text_box_position[2][0]
                y1 = text_box_position[2][1]
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                if 1080 < cx and cx < 1270 and 30 < cy and cy < 80:
                    cur_huobi_str = text
                if 660 < cx and cx < 780 and 300 < cy and cy < 350:
                    duizhangji1 = text
                if 660 < cx and cx < 780 and 500 < cy and cy < 550:
                    duizhangji2 = text
            logger.info(cur_huobi_str)
            logger.info(duizhangji1)
            logger.info(duizhangji2)
            time.sleep(3.0)
            cur_huobi = 0
            max_huobi = 99999
            text = cur_huobi_str
            if cur_huobi_str != '' or cur_huobi_str is not None:
                text_len = len(cur_huobi_str)
                end_ = text_len - 5
                if end_ > 0:
                    if text[:end_].isdigit():
                        cur_huobi = int(text[:end_])
                    else:
                        cur_huobi = 9999
                    if text[end_+1:].isdigit():
                        max_huobi = int(text[end_+1:])
            logger.info(cur_huobi)
            logger.info(max_huobi)
            tg = max_huobi*2
            if shuamanjiuting:
                tg = max_huobi
            if cur_huobi >= tg:  # ???????????????
                break_program = True
            else:  # ?????????????????????
                # ????????????
                mouse_click(xbase + 130, ybase + 370)  # ?????????
                time.sleep(t_1)

                # ??????????????????
                now = dt.datetime.now()
                fname = now.strftime("%Y-%m-%d_%H-%M-%S")
                autopy.bitmap.capture_screen().save('D://%s/temp.png' % (dir_name,))
                img = cv2.imread('D://%s/temp.png' % (dir_name,))
                screen_shot = img[ybase:yend, xbase:xend, ...]
                np_images = [screen_shot]
                results = ocr.recognize_text(
                    images=np_images,  # ???????????????ndarray.shape ??? [H, W, C]???BGR?????????
                    use_gpu=use_gpu,  # ???????????? GPU????????????GPU???????????????CUDA_VISIBLE_DEVICES????????????
                    output_dir=output_dir,  # ???????????????????????????????????? ocr_result???
                    visualization=True,  # ?????????????????????????????????????????????
                    box_thresh=0.5,  # ????????????????????????????????????
                    text_thresh=0.5)  # ???????????????????????????????????????
                data = results[0]['data']
                cur_char = ''
                for d in data:
                    text = d[0]
                    text_box_position = d[1]
                    x0 = text_box_position[0][0]
                    y0 = text_box_position[0][1]
                    x1 = text_box_position[2][0]
                    y1 = text_box_position[2][1]
                    cx = (x0 + x1) / 2
                    cy = (y0 + y1) / 2
                    if 180 < cx and cx < 310 and 250 < cy and cy < 290:
                        cur_char = text
                        break
                logger.info('???????????????%s' % cur_char)
                if juese_name in cur_char:
                    mouse_click(xbase + 60, ybase + 60)  # ??????
                    time.sleep(t_1)
                else:
                    mouse_click(xbase + 20, ybase + 440)  # ??????
                    time.sleep(t_1)
                    mouse_click(xbase + ziji_coord[0], ybase + ziji_coord[1])
                    time.sleep(t_1)
                    mouse_click(xbase + 1140, ybase + 700)  # ??????
                    time.sleep(t_1)
                    mouse_click(xbase + 100, ybase + 600)  # ????????????
                    time.sleep(t_1)
                    mouse_click(xbase + 1140, ybase + 700)  # ??????
                    time.sleep(t_1)

                # ???????????????
                if duiyou_ji[duiyou_2] not in duizhangji2:
                    mouse_click(xbase + 500, ybase + 370)  # ?????????2
                    time.sleep(t_1)
                    mouse_click(xbase + 1140, ybase + 700)  # ??????
                    time.sleep(t_1)
                if duiyou_ji[duiyou_1] not in duizhangji1:
                    mouse_click(xbase + 310, ybase + 370)  # ?????????1
                    time.sleep(t_1)
                    mouse_click(xbase + 1140, ybase + 700)  # ??????
                    time.sleep(t_1)

                if duiyou_ji[duiyou_1] not in duizhangji1:
                    mouse_click(xbase + 310, ybase + 370)  # ?????????1
                    time.sleep(t_1)
                    mouse_click(xbase + duiyou_xy[duiyou_1][0], ybase + duiyou_xy[duiyou_1][1])
                    time.sleep(t_1)
                    mouse_click(xbase + 1140, ybase + 700)  # ??????
                    time.sleep(t_1)
                if duiyou_ji[duiyou_2] not in duizhangji2:
                    mouse_click(xbase + 500, ybase + 370)  # ?????????2
                    time.sleep(t_1)
                    mouse_click(xbase + duiyou_xy[duiyou_2][0], ybase + duiyou_xy[duiyou_2][1])
                    time.sleep(t_1)
                    mouse_click(xbase + 1140, ybase + 700)  # ??????
                    time.sleep(t_1)


                mouse_click(xbase + 1140, ybase + 700)  # ????????????
                time.sleep(t_1)
                if nandu == 0:
                    mouse_click(xbase + 300, ybase + 230)  # ????????????
                elif nandu == 1:
                    mouse_click(xbase + 300, ybase + 320)  # ????????????
                elif nandu == 2:
                    mouse_click(xbase + 300, ybase + 410)  # ????????????
                elif nandu == 3:
                    mouse_click(xbase + 300, ybase + 490)  # ????????????
                elif nandu == 4:
                    mouse_click(xbase + 300, ybase + 490)  # ????????????110
                    time.sleep(0.1)
                    mouse_click(xbase + 1200, ybase + 570)
                    time.sleep(0.1)
                    mouse_click(xbase + 1200, ybase + 480)
                    time.sleep(0.1)
                    mouse_click(xbase + 1200, ybase + 380)
                    time.sleep(0.1)
                    mouse_click(xbase + 1200, ybase + 280)
                    time.sleep(0.1)
                elif nandu == 5:
                    mouse_click(xbase + 300, ybase + 490)  # ????????????130
                    time.sleep(0.1)
                    mouse_click(xbase + 1200, ybase + 570)
                    time.sleep(0.1)
                    mouse_click(xbase + 1200, ybase + 480)
                    time.sleep(0.1)
                    mouse_click(xbase + 1200, ybase + 380)
                    time.sleep(0.1)
                    mouse_click(xbase + 1200, ybase + 280)
                    time.sleep(0.1)
                    key_long_press('up', t=0.1)
                    time.sleep(1.5)
                    mouse_click(xbase + 1200, ybase + 240)
                    time.sleep(0.1)
                    key_long_press('up', t=0.5)
                    time.sleep(1.5)
                    mouse_click(xbase + 1200, ybase + 240)
                    time.sleep(0.1)
                    if shanghaileixing == '??????':
                        mouse_click(xbase + 1200, ybase + 340)  # ???????????????????????????15%
                        time.sleep(0.1)
                    else:
                        mouse_click(xbase + 1200, ybase + 440)  # ???????????????????????????15%
                        time.sleep(0.1)
                time.sleep(t_1)
                mouse_click(xbase + 1140, ybase + 700)  # ????????????
                time.sleep(t_1)
                mouse_click(xbase + 810, ybase + 530)  # ??????
        elif here == 'duihua':
            logger.info('duihua')
            mouse_click(xbase + 1200, ybase + 65)  # ??????
            time.sleep(t_1)
            mouse_click(xbase + 810, ybase + 500)  # ??????
            time.sleep(2.3)
        elif here == 'xuanzekeyin':
            logger.info('xuanzekeyin')
            # logger.info(data)
            # time.sleep(1200.0)
            logger.info(keyin_id)
            if keyin_id == 0 and defen < 1:   # ????????????????????????
                logger.info('????????????????????????')
                manyi = 0
                for d in data:
                    text = d[0]
                    if manyis[0] in text or manyis[1] in text:
                        text_box_position = d[1]
                        x0 = text_box_position[0][0]
                        y0 = text_box_position[0][1]
                        x1 = text_box_position[2][0]
                        y1 = text_box_position[2][1]
                        cx = (x0 + x1) / 2
                        cy = (y0 + y1) / 2
                        keyin_list.append(text)
                        mouse_click(xbase + cx, ybase + cy)  # ????????????
                        time.sleep(t_1)
                        mouse_click(xbase + 1130, ybase + 690)  # ????????????
                        time.sleep(t_1)
                        manyi = 1
                        break
                if manyi == 0:
                    logger.info('bumanyi')
                    first_keyin = ''
                    for d in data:
                        text = d['text']
                        text_box_position = d['text_box_position']
                        x0 = text_box_position[0][0]
                        y0 = text_box_position[0][1]
                        x1 = text_box_position[2][0]
                        y1 = text_box_position[2][1]
                        cx = (x0 + x1) / 2
                        cy = (y0 + y1) / 2
                        if 653 < cx and cx < 1202 and 247 < cy and cy < 270:
                            first_keyin = text
                            break
                    keyin_list.append(first_keyin)
                    mouse_click(xbase + 1130, ybase + 270)  # ????????????
                    time.sleep(t_1)
                    mouse_click(xbase + 1130, ybase + 690)  # ????????????
                    time.sleep(t_1)
                    mouse_click(xbase + 50, ybase + 50)  # ??????
                    time.sleep(t_1)
                    mouse_click(xbase + 770, ybase + 688)  # ??????
                    time.sleep(t_1)
                    mouse_click(xbase + 470, ybase + 530)  # ??????
                    time.sleep(1.8)
                    mouse_click(xbase + 470, ybase + 530)  # ??????
                    time.sleep(t_1)
            else:
                manyi = 0
                for mmm in manyis:
                    for d in data:
                        text = d['text']
                        if mmm in text:
                            text_box_position = d['text_box_position']
                            x0 = text_box_position[0][0]
                            y0 = text_box_position[0][1]
                            x1 = text_box_position[2][0]
                            y1 = text_box_position[2][1]
                            cx = (x0 + x1) / 2
                            cy = (y0 + y1) / 2
                            keyin_list.append(text)
                            mouse_click(xbase + cx, ybase + cy)  # ????????????
                            time.sleep(t_1)
                            mouse_click(xbase + 1130, ybase + 690)  # ????????????
                            time.sleep(t_1)
                            manyi = 1
                            break
                    if manyi == 1:
                        break
                if manyi == 0:
                    logger.info('bumanyi')
                    first_keyin = ''
                    for d in data:
                        text = d['text']
                        text_box_position = d['text_box_position']
                        x0 = text_box_position[0][0]
                        y0 = text_box_position[0][1]
                        x1 = text_box_position[2][0]
                        y1 = text_box_position[2][1]
                        cx = (x0 + x1) / 2
                        cy = (y0 + y1) / 2
                        if 653 < cx and cx < 1202 and 247 < cy and cy < 270:
                            first_keyin = text
                            break
                    keyin_list.append(first_keyin)
                    mouse_click(xbase + 1130, ybase + 270)  # ????????????
                    time.sleep(t_1)
                    mouse_click(xbase + 1130, ybase + 690)  # ????????????
                    time.sleep(t_1)
            keyin_id += 1
            no_maodian_count = 0
            zhandou_end = True
        elif here == 'zhandou':
            # ??????????????????
            defen_str = '0'
            for d in data:
                text = d['text']
                text_box_position = d['text_box_position']
                x0 = text_box_position[0][0]
                y0 = text_box_position[0][1]
                x1 = text_box_position[2][0]
                y1 = text_box_position[2][1]
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                if 75 < cx and cx < 186 and 108 < cy and cy < 143:
                    defen_str = text
                    break
            if '??????' in defen_str:
                defen_str = defen_str[2:]
            if defen_str.isdigit():
                defen = int(defen_str)


            if zhandou_end:
                # ????????????????????????????????????
                # logger.info('zhandou end')

                mouse_click(xbase + 1010, ybase + 620)  # ??????????????????
                time.sleep(0.01)

                zaimaodian = 0
                for d in data:
                    text = d['text']
                    if '??????' in text:
                        text_box_position = d['text_box_position']
                        x0 = text_box_position[0][0]
                        y0 = text_box_position[0][1]
                        x1 = text_box_position[2][0]
                        y1 = text_box_position[2][1]
                        cx = (x0 + x1) / 2
                        cy = (y0 + y1) / 2
                        if 60 < cx and cx < 205 and 270 < cy and cy < 300:
                            zaimaodian = 1
                            break
                if zaimaodian == 0:
                    image = screen_shot
                    pimage, im_size = _decode.process_image(np.copy(image))
                    image, boxes, scores, classes = _decode.detect_image(image, pimage, im_size, draw_image, draw_thresh)
                    # filename = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
                    # if draw_image:
                    #     cv2.imwrite('images/res/%s.png' % filename, image)
                    if len(scores) == 0:
                        logger.info('no objs.')
                        rrrrrr = random.choice(range(3))
                        if rrrrrr == 0:
                            key_long_press('<-', t=0.4)
                        if rrrrrr == 1:
                            pass
                        if rrrrrr == 2:
                            key_long_press('->', t=0.4)
                        no_maodian_count += 1
                        rrrrrr = random.choice(range(2))
                        t_c = 2
                        if rrrrrr == 0:
                            t_c = 2
                        if rrrrrr == 1:
                            t_c = 6
                        if no_maodian_count > 3:
                            for pp in range(t_c):
                                key_long_press('w', t=0.5)
                                if pp < t_c-1:
                                    # ??????????????????
                                    now = dt.datetime.now()
                                    fname = now.strftime("%Y-%m-%d_%H-%M-%S")
                                    autopy.bitmap.capture_screen().save('D://%s/temp.png' % (dir_name,))
                                    img = cv2.imread('D://%s/temp.png' % (dir_name,))
                                    screen_shot = img[ybase:yend, xbase:xend, ...]
                                    image = screen_shot
                                    pimage, im_size = _decode.process_image(np.copy(image))
                                    image, boxes, scores, classes = _decode.detect_image(image, pimage, im_size, draw_image, draw_thresh)
                                    if len(scores) != 0:
                                        logger.info('?????????????????????!')
                                        break
                        time.sleep(0.01)
                    else:
                        logger.info('?????????????????????!')
                        pp = 0
                        x0 = 0.0
                        y0 = 0.0
                        x1 = 0.0
                        y1 = 0.0
                        min_chajv_x = 99999.0
                        bcx = (x0 + x1) / 2.0
                        bcy = (y0 + y1) / 2.0
                        for tt in range(len(classes)):
                            cl = classes[tt]
                            x0 = boxes[tt][0]
                            y0 = boxes[tt][1]
                            x1 = boxes[tt][2]
                            y1 = boxes[tt][3]
                            bcx = (x0 + x1) / 2.0
                            bcy = (y0 + y1) / 2.0
                            chajv_x = bcx - 640
                            chajv_x = abs(chajv_x)
                            if cl == 1:
                                no_maodian_count = 0
                                if min_chajv_x > chajv_x:
                                    min_chajv_x = chajv_x
                                    pp = tt
                        x0 = boxes[pp][0]
                        y0 = boxes[pp][1]
                        x1 = boxes[pp][2]
                        y1 = boxes[pp][3]
                        cid = classes[pp]
                        logger.info('??????%s????????????!' % class_cn_names[cid])
                        bcx = (x0 + x1) / 2.0
                        bcy = (y0 + y1) / 2.0
                        chajv_x = bcx - 640
                        chajv_y = 580 - bcy
                        # tan = chajv_y / chajv_x
                        # tan = abs(tan)
                        press_key = '->'
                        if chajv_x > 0:
                            press_key = '->'
                        else:
                            press_key = '<-'
                        ddd = abs(chajv_x)
                        t_2 = 0.01
                        if 0 <= ddd and ddd < 100:
                            t_2 = 0.001
                        if 100 <= ddd and ddd < 400:
                            t_2 = 0.1
                        if 400 <= ddd and ddd < 10000:
                            t_2 = 0.2
                        key_long_press(press_key, t=t_2)
                        time.sleep(0.1)
                        key_long_press('w', t=0.5)
                        time.sleep(0.3)
                    mouse_click(xbase + 1100, ybase + 680)  # ?????????
                    time.sleep(0.3)
                else:  # ???????????????
                    mouse_click(xbase + 1100, ybase + 680)  # ?????????
                    time.sleep(0.3)

                mouse_click(xbase + 1220, ybase + 680)  # ?????????
                time.sleep(1.3)
                # ??????????????????????????????????????????????????????
                no_enemy = True
                for d in data:
                    text = d['text']
                    text_box_position = d['text_box_position']
                    x0 = text_box_position[0][0]
                    y0 = text_box_position[0][1]
                    x1 = text_box_position[2][0]
                    y1 = text_box_position[2][1]
                    cx = (x0 + x1) / 2
                    cy = (y0 + y1) / 2
                    if 730 < cx and cx < 860 and 30 < cy and cy < 110:
                        logger.info('??????????????????%s' % text)
                        no_enemy = False
                        break
                if not no_enemy:
                    no_enemy_count = 0
                    zhandou_end = False
            else:
                # logger.info('zhandou ing...')
                if attack_count < 10:
                    mouse_click(xbase + 1100, ybase + 470)  # ??????f
                    time.sleep(0.01)

                # ???????????????sp
                cur_hp = 0
                cur_hp_str = ''
                cur_sp_str = ''
                cur_sp = 0
                data = results[0]['data']
                for d in data:
                    text = d['text']
                    text_box_position = d['text_box_position']
                    x0 = text_box_position[0][0]
                    y0 = text_box_position[0][1]
                    x1 = text_box_position[2][0]
                    y1 = text_box_position[2][1]
                    cx = (x0 + x1) / 2
                    cy = (y0 + y1) / 2
                    # ??????
                    if 377 < cx and cx < 468 and 667 < cy and cy < 694:
                        cur_hp_str = text
                        if text != '' or text is not None:
                            text_len = len(text)
                            end_ = text_len - 5
                            if end_ > 0:
                                if text[:end_].isdigit():
                                    cur_hp = int(text[:end_])
                                else:
                                    cur_hp = 99999
                    # sp
                    if 771 < cx and cx < 833 and 718 < cy and cy < 738:
                        cur_sp_str = text
                        if text != '' or text is not None:
                            text_len = len(text)
                            end_ = text_len - 4
                            if end_ > 0:
                                if text[:end_].isdigit():
                                    cur_sp = int(text[:end_])
                                else:
                                    cur_sp = 999

                # ??????????????????????????????
                if chara == 0 or chara == 4:
                    if attack_count > 5:
                        mouse_click(xbase + 1220, ybase + 430)  # ??????2????????????
                        time.sleep(0.01)
                        mouse_click(xbase + 1220, ybase + 330)  # ??????1????????????
                        time.sleep(0.01)
                    if cur_hp < 4100:
                        mouse_click(xbase + 1010, ybase + 620)  # ??????????????????
                        time.sleep(0.01)

                    if attack_count > bisha:
                        mouse_click(xbase + 1220, ybase + 570)  # ?????????
                    mouse_click(xbase + 1220, ybase + 680)  # ?????????
                    if attack_count % shangbijiange == 0:
                        rrrrrr = random.choice(range(2))
                        fangxiang_key = 'a'
                        if rrrrrr == 0:
                            fangxiang_key = 'a'
                        if rrrrrr == 1:
                            fangxiang_key = 'd'
                        key_press(fangxiang_key)
                        mouse_click(xbase + 1100, ybase + 680)  # ????????????
                        key_release(fangxiang_key)
                        time.sleep(1.0)
                        mouse_long_press(xbase + 1220, ybase + 680, t=1.0)  # ???????????????????????????
                        time.sleep(0.3)
                    if attack_count % xuli == 0:
                        time.sleep(0.3)
                        mouse_long_press(xbase + 1220, ybase + 680, t=1.0)  # ???????????????????????????
                        time.sleep(0.3)
                elif chara == 1:
                    if attack_count > 5:
                        mouse_click(xbase + 1220, ybase + 430)  # ??????2????????????
                        time.sleep(0.01)
                        mouse_click(xbase + 1220, ybase + 330)  # ??????1????????????
                        time.sleep(0.01)
                    if cur_hp < 4100:
                        mouse_click(xbase + 1010, ybase + 620)  # ??????????????????
                        time.sleep(0.01)

                    jiange = time.time() - dazhao_last_time  # ????????????????????????????????????jiange???
                    if attack_count > bisha and jiange > 18 and cur_sp > 125:
                        logger.info('?????????')
                        mouse_click(xbase + 1220, ybase + 680)  # ?????????
                        time.sleep(0.5)
                        for _ in range(10):
                            mouse_click(xbase + 1220, ybase + 570)  # ?????????
                            time.sleep(0.15)
                        time.sleep(3.8)
                        dazhao_last_time = time.time()  # ????????????????????????
                        baofa_ing = True
                    jiange = time.time() - dazhao_last_time  # ????????????????????????????????????jiange???
                    if jiange > 18:
                        baofa_ing = False

                    if baofa_ing:
                        logger.info('?????????')
                        mouse_click(xbase + 1100, ybase + 570)  # ????????????
                        time.sleep(0.01)
                        mouse_click(xbase + 1220, ybase + 680)  # ?????????
                        time.sleep(0.3)
                        mouse_click(xbase + 1100, ybase + 680)  # ????????????
                    else:
                        logger.info('????????????')
                        if attack_count == 2:
                            mouse_click(xbase + 1100, ybase + 680)  # ????????????
                            time.sleep(0.3)
                            mouse_long_press(xbase + 1220, ybase + 680, t=1.0)  # ???????????????????????????
                            time.sleep(0.01)
                        mouse_click(xbase + 1220, ybase + 680)  # ?????????
                        time.sleep(0.01)
                        if attack_count % xuli == 0:
                            mouse_long_press(xbase + 1220, ybase + 680, t=1.0)  # ???????????????????????????
                            time.sleep(0.01)
                elif chara == 2:
                    if attack_count > 5:
                        mouse_click(xbase + 1220, ybase + 430)  # ??????2????????????
                        time.sleep(0.01)
                        mouse_click(xbase + 1220, ybase + 330)  # ??????1????????????
                        time.sleep(0.01)
                    if cur_hp < 4100:
                        mouse_click(xbase + 1010, ybase + 620)  # ??????????????????
                        time.sleep(0.01)

                    jiange = time.time() - dazhao_last_time  # ????????????????????????????????????jiange???
                    if attack_count > bisha and jiange > 22:
                        logger.info('?????????????????????')
                        mouse_click(xbase + 1220, ybase + 570)  # ??????????????????
                        time.sleep(1.0)
                        if cur_sp > 75:
                            mouse_long_press(xbase + 1220, ybase + 570, t=1.0)  # ???????????????
                            logger.info('..............???????????????')
                            time.sleep(4.2)
                        mouse_long_press(xbase + 1220, ybase + 680, t=1.0)  # ???????????????
                        dazhao_last_time = time.time()  # ????????????????????????
                        time.sleep(0.5)
                        mouse_click(xbase + 1100, ybase + 570)  # ????????????
                        time.sleep(1.3)
                        mouse_click(xbase + 1100, ybase + 680)  # ????????????
                        time.sleep(0.1)
                        baofa_ing = True
                    jiange = time.time() - dazhao_last_time  # ????????????????????????????????????jiange???
                    if jiange > 12:
                        baofa_ing = False

                    if baofa_ing:
                        logger.info('?????????')
                        mouse_click(xbase + 1220, ybase + 680)  # ?????????
                        time.sleep(0.01)
                        jiange = time.time() - dazhao_last_time  # ????????????????????????????????????jiange???
                        if 2.5 < jiange and jiange < 3.5:
                            logger.info('????????????????????????...')
                            mouse_long_press(xbase + 1220, ybase + 680, t=1.0)  # ???????????????
                        if 8.7 < jiange and jiange < 9.5:
                            logger.info('????????????????????????...')
                            mouse_long_press(xbase + 1220, ybase + 680, t=1.0)  # ???????????????
                    else:
                        logger.info('????????????')
                        mouse_click(xbase + 1220, ybase + 680)  # ?????????
                        jiange = time.time() - dazhao_last_time  # ????????????????????????????????????jiange???
                        if 13.5 < jiange and jiange < 14.5:
                            logger.info('?????????????????????????????????')
                            mouse_click(xbase + 1220, ybase + 570)  # ??????????????????
                            time.sleep(0.5)
                        time.sleep(0.01)
                elif chara == 3:
                    if attack_count > 5:
                        mouse_click(xbase + 1220, ybase + 430)  # ??????2????????????
                        time.sleep(0.01)
                        mouse_click(xbase + 1220, ybase + 330)  # ??????1????????????
                        time.sleep(0.01)
                    if cur_hp < 4100:
                        mouse_click(xbase + 1010, ybase + 620)  # ??????????????????
                        time.sleep(0.01)

                    jiange = time.time() - dazhao_last_time  # ????????????????????????????????????jiange???
                    if attack_count > bisha and jiange > 20:
                        logger.info('??????????????????')
                        mouse_long_press(xbase + 1100, ybase + 570, t=1.0)  # ???????????????
                        time.sleep(1.0)
                        mouse_click(xbase + 1220, ybase + 680)  # ?????????
                        time.sleep(1.9)
                        if cur_sp > 100:
                            for _ in range(3):
                                mouse_click(xbase + 1220, ybase + 570)  # ?????????
                            logger.info('..............???????????????')
                            time.sleep(3.0)
                        dazhao_last_time = time.time()  # ????????????????????????
                        baofa_ing = True
                    jiange = time.time() - dazhao_last_time  # ????????????????????????????????????jiange???
                    if jiange > 22:
                        baofa_ing = False

                    if baofa_ing:
                        logger.info('?????????')
                        mouse_click(xbase + 1220, ybase + 680)  # ?????????
                        time.sleep(0.01)
                        jiange = time.time() - dazhao_last_time  # ????????????????????????????????????jiange???
                        if 1.0 < jiange and jiange < 2.0:
                            logger.info('????????????????????????...')
                            mouse_long_press(xbase + 1220, ybase + 680, t=1.0)  # ???????????????
                        if 6.5 < jiange and jiange < 7.5:
                            logger.info('????????????????????????...')
                            mouse_long_press(xbase + 1220, ybase + 680, t=1.0)  # ???????????????
                        if 12.0 < jiange and jiange < 13.0:
                            logger.info('????????????????????????...')
                            mouse_long_press(xbase + 1220, ybase + 680, t=1.0)  # ???????????????
                        if 17.5 < jiange and jiange < 18.5:
                            logger.info('????????????????????????...')
                            mouse_long_press(xbase + 1220, ybase + 680, t=1.0)  # ???????????????
                    else:
                        logger.info('????????????')
                        mouse_click(xbase + 1220, ybase + 680)  # ?????????
                        time.sleep(0.01)

                # ???????????????????????????????????????????????????
                if attack_count > 5:
                    no_enemy = True
                    for d in data:
                        text = d['text']
                        text_box_position = d['text_box_position']
                        x0 = text_box_position[0][0]
                        y0 = text_box_position[0][1]
                        x1 = text_box_position[2][0]
                        y1 = text_box_position[2][1]
                        cx = (x0 + x1) / 2
                        cy = (y0 + y1) / 2
                        if 730 < cx and cx < 860 and 30 < cy and cy < 110:
                            logger.info('??????????????????%s???HP:%d  SP:%d' % (text, cur_hp, cur_sp))
                            no_enemy = False
                            break
                    if no_enemy:
                        no_enemy_count += 1
                        time.sleep(0.5)
                    else:
                        no_enemy_count = 0
                    if no_enemy_count > 8:
                        zhandou_end = True
                        no_maodian_count = 0

                time.sleep(0.01)
                logger.info('attack_count: %d' % attack_count)
                attack_count += 1
        elif here == 'jiazai':
            logger.info('jiazai')
            attack_count = 0
            no_enemy_count = 0
            qian2 = 0
            for pp in range(len(keyin_list)):
                ky = keyin_list[pp]
                logger.info('%d: %s'%(pp, ky))
                if manyis[0] in ky:
                    qian2 += 1
                if manyis[1] in ky:
                    qian2 += 1
            if qian2 == 2:
                logger.info('????????????2?????????buff.')
            elif qian2 == 1:
                logger.info('????????????1?????????buff.')
            jiange = time.time() - jiaoben_start_time  # ?????????????????????jiange???
            jiange = int(jiange)
            fenzhong = jiange // 60
            miaozhong = jiange % 60
            logger.info('??????????????????%d???%d??????'%(fenzhong, miaozhong))
            zhandou_end = False
            time.sleep(1.0)
        elif here == 'unknow':
            logger.info('unknow')
            mouse_click(xbase + 640, ybase + 440)  # ??????
            time.sleep(0.01)
        elif here == 'zhandouzhunbei2':
            logger.info('zhandouzhunbei2')
            mouse_click(xbase + 240, ybase + 60)  # ???????????????
        elif here == 'letu_jiesuan':
            logger.info('letu_jiesuan')
            mouse_click(xbase + 640, ybase + 440)  # ??????
            time.sleep(1.0)
        elif here == 'letu_tcjiesuan':
            logger.info('letu_tcjiesuan')
            mouse_click(xbase + 640, ybase + 440)  # ??????
            time.sleep(0.3)
            mouse_click(xbase + 640, ybase + 690)  # mvp??????????????????
            time.sleep(2.0)
        elif here == 'feilisishangdian':
            logger.info('feilisishangdian')
            mouse_click(xbase + 1100, ybase + 125)  # ????????????
            time.sleep(t_1)
            mouse_click(xbase + 900, ybase + 300)  # ??????
            time.sleep(t_1)
            mouse_click(xbase + 1140, ybase + 700)
            time.sleep(t_1)
            mouse_click(xbase + 1140, ybase + 700)
            time.sleep(t_1)
            mouse_click(xbase + 1140, ybase + 700)
            time.sleep(t_1)
            mouse_click(xbase + 300, ybase + 300)  # ??????
            time.sleep(t_1)
            mouse_click(xbase + 1140, ybase + 700)
            time.sleep(t_1)
            mouse_click(xbase + 1140, ybase + 700)
            time.sleep(t_1)
            mouse_click(xbase + 1140, ybase + 700)
            time.sleep(t_1)
            mouse_click(xbase + 66, ybase + 66)  # ??????
            time.sleep(t_1)
            key_long_press('s', t=1.0)
            time.sleep(t_1)
            key_long_press('a', t=1.0)
            time.sleep(t_1)
        elif here == 'gerenxinxi':
            mouse_click(xbase + 240, ybase + 60)  # ???????????????
            time.sleep(1.0)
            logger.info('gerenxinxi')
        time.sleep(0.01)
        # ch = cv2.waitKey(0)
        # if ch == 27 or ch == ord("q") or ch == ord("Q"):
        #     break


@logger.catch
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

    setup_logger(file_name, distributed_rank=0, filename="game_log.txt", mode="o")
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
    elif archi_name == 'PPYOLOE':
        # PPYOLOE????????????multiclass_nms?????????multiclass_nms????????????
        if args.conf is not None:
            exp.nms_cfg['score_threshold'] = args.conf
        if args.tsize is not None:
            exp.test_size = [args.tsize, args.tsize]
            exp.head['eval_size'] = exp.test_size
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
        pass
    elif archi_name == 'PPYOLO':
        pass
    elif archi_name == 'PPYOLOE':
        # ??????????????????
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

        predictor = PPYOLOEPredictor(
            model, exp, trt_file,
            args.device, args.fp16, args.legacy,
        )
    elif archi_name == 'FCOS':
        pass
    else:
        raise NotImplementedError("Architectures \'{}\' is not implemented.".format(archi_name))
    current_time = time.localtime()
    image_demo(predictor, vis_folder, args.path, current_time, args.save_result, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    # ???????????????????????????
    isDebug = True if sys.gettrace() else False
    if isDebug:
        print('Debug Mode.')
        args.exp_file = '../' + args.exp_file
        args.ckpt = '../' + args.ckpt   # ?????????????????????????????????????????????
        args.path = '../' + args.path   # ?????????????????????????????????????????????
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
