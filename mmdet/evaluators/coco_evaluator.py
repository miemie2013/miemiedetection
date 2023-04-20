#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import tempfile
import time
import numpy as np
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm
import pycocotools.mask as maskUtils
import torch

from mmdet.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)


def per_class_AR_table(coco_eval, class_names, headers=["class", "AR"], colums=6):
    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    for idx, name in enumerate(class_names):
        recall = recalls[:, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100)

    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


def per_class_AP_table(coco_eval, class_names, headers=["class", "AP"], colums=6):
    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


class COCOEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, dataloader, img_size, confthre, nmsthre, num_classes, archi_name='', testdev=False, per_class_AP=True, per_class_AR=True
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.archi_name = archi_name
        self.testdev = testdev
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR

    def evaluate_yolox(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

            data_list.extend(self.convert_to_coco_format(outputs, info_imgs, ids))

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_ppyolo(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        test_size=None,
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        data_list = []
        progress_bar = iter if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)
        steps = len(self.dataloader)
        print_interval = steps // 10
        num_imgs = self.dataloader.dataset.num_record

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        eval_start = time.time()
        for cur_iter, (pimages, im_sizes, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                pimages = pimages.type(tensor_type)     # [N, 3, 608, 608]
                im_sizes = im_sizes.type(tensor_type)   # [N, 2]

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                preds = model(pimages, im_sizes)

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start
                # NMS包含在了模型里，这里记录无效...
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end
                if cur_iter % print_interval == 0:
                    progress_str = "Eval iter: {}/{}".format(cur_iter + 1, steps)
                    logger.info(progress_str)

            data_list.extend(self.convert_to_coco_format2(preds, ids))
        cost = time.time() - eval_start
        logger.info('Eval time: %.1f s;  Speed: %.1f FPS.'%(cost, (num_imgs / cost)))

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_ppyoloe(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        test_size=None,
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        data_list = []
        progress_bar = iter if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)
        steps = len(self.dataloader)
        print_interval = steps // 10
        num_imgs = self.dataloader.dataset.num_record

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        eval_start = time.time()
        for cur_iter, (pimages, scale_factor, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                pimages = pimages.type(tensor_type)     # [N, 3, 640, 640]
                scale_factor = scale_factor.type(tensor_type)   # [N, 2]

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                preds = model(pimages, scale_factor)

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start
                # NMS包含在了模型里，这里记录无效...
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end
                if cur_iter % print_interval == 0:
                    progress_str = "Eval iter: {}/{}".format(cur_iter + 1, steps)
                    logger.info(progress_str)

            data_list.extend(self.convert_to_coco_format3(preds, ids))
        cost = time.time() - eval_start
        logger.info('Eval time: %.1f s;  Speed: %.1f FPS.'%(cost, (num_imgs / cost)))

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_solo(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        test_size=None,
    ):
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        bbox_list = []
        mask_list = []
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        for cur_iter, (pimages, im_sizes, ori_shapes, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # 方案2，用SOLOv2Pad，需要多余的预处理。
                # max_hw, _ = im_sizes.max(0)
                # max_hw = max_hw.cpu().detach().numpy()
                # max_h = max_hw[0]
                # max_w = max_hw[1]
                # coarsest_stride = 32
                # max_h = int(np.ceil(max_h / coarsest_stride) * coarsest_stride)
                # max_w = int(np.ceil(max_w / coarsest_stride) * coarsest_stride)
                # pimages = pimages[:, :, :max_h, :max_w]

                pimages = pimages.type(tensor_type)     # [N, 3, 800, 1xxx]
                im_sizes = im_sizes.type(tensor_type)   # [N, 2]
                ori_shapes = ori_shapes.type(tensor_type)   # [N, 2]

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                preds = model(pimages, im_sizes, ori_shapes)

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start
                # NMS包含在了模型里，这里记录无效...
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

            bbox_data, mask_data = self.convert_to_coco_format_solo(preds, ids)
            bbox_list.extend(bbox_data)
            mask_list.extend(mask_data)

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            bbox_list = gather(bbox_list, dst=0)
            bbox_list = list(itertools.chain(*bbox_list))
            mask_list = gather(mask_list, dst=0)
            mask_list = list(itertools.chain(*mask_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction_solo(bbox_list, mask_list, statistics)
        synchronize()
        return eval_results

    def evaluate_fcos(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        test_size=None,
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        data_list = []
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        for cur_iter, (pimages, im_scales, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                pimages = pimages.type(tensor_type)     # [N, 3, H, W]
                im_scales = im_scales.type(tensor_type)   # [N, 1]

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                preds = model(pimages, im_scales)

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start
                # NMS包含在了模型里，这里记录无效...
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

            data_list.extend(self.convert_to_coco_format3(preds, ids))

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def convert_to_coco_format2(self, preds, ids):
        data_list = []
        for (pred, img_id) in zip(
            preds, ids
        ):
            if pred is None:
                continue
            if pred[0][0] < 0.0:
                continue
            output = pred.cpu()

            bboxes = output[:, 2:6]  # xyxy
            # 这里开始YOLOX和PPYOLO的评测代码不同
            # bboxes = xyxy2xywh(bboxes)

            cls = output[:, 0]
            scores = output[:, 1]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.clsid2catid[int(cls[ind])]

                # YOLOX和PPYOLO的评测代码不同
                xmin, ymin, xmax, ymax = bboxes[ind].numpy().tolist()
                # PPYOLO需要+1，YOLOX不需要+1
                w = xmax - xmin + 1
                h = ymax - ymin + 1
                # w = xmax - xmin
                # h = ymax - ymin
                bbox = [xmin, ymin, w, h]
                # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
                bbox = [round(float(x) * 10) / 10 for x in bbox]

                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    # "bbox": bboxes[ind].numpy().tolist(),
                    "bbox": bbox,
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def convert_to_coco_format3(self, preds, ids):
        data_list = []
        for (pred, img_id) in zip(
            preds, ids
        ):
            if pred is None:
                continue
            if pred[0][0] < 0.0:
                continue
            output = pred.cpu()

            bboxes = output[:, 2:6]  # xyxy
            # 这里开始YOLOX和PPYOLO的评测代码不同
            # bboxes = xyxy2xywh(bboxes)

            cls = output[:, 0]
            scores = output[:, 1]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.clsid2catid[int(cls[ind])]

                # YOLOX和PPYOLO的评测代码不同
                xmin, ymin, xmax, ymax = bboxes[ind].numpy().tolist()
                # 不需要+1
                # w = xmax - xmin + 1
                # h = ymax - ymin + 1
                w = xmax - xmin
                h = ymax - ymin
                bbox = [xmin, ymin, w, h]
                # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
                bbox = [round(float(x) * 10) / 10 for x in bbox]

                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    # "bbox": bboxes[ind].numpy().tolist(),
                    "bbox": bbox,
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def convert_to_coco_format_solo(self, preds, ids):
        bbox_data = []
        mask_data = []
        for k, img_id in enumerate(ids):
            bbox_num = preds['bbox_num'][k]
            if bbox_num == 0:
                continue

            masks = preds['segm'][k]
            cls = preds['cate_label'][k]
            scores = preds['cate_score'][k]

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

            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.clsid2catid[int(cls[ind])]

                xmin, ymin, xmax, ymax = bboxes[ind].tolist()
                # 需不需要+1？
                w = xmax - xmin + 1
                h = ymax - ymin + 1
                # w = xmax - xmin
                # h = ymax - ymin
                bbox = [xmin, ymin, w, h]
                # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
                bbox = [round(float(x) * 10) / 10 for x in bbox]

                bbox_res = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bbox,
                    "score": float(scores[ind]),
                }
                bbox_data.append(bbox_res)

                segm = maskUtils.encode(np.asfortranarray(masks[ind].astype(np.uint8)))
                segm['counts'] = segm['counts'].decode('utf8')
                mask_res = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "segmentation": segm,
                    "score": float(scores[ind]),
                }
                mask_data.append(mask_res)
        return bbox_data, mask_data

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            if self.testdev:
                json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, "w"))
                cocoDt = cocoGt.loadRes(tmp)
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval

                logger.warning("Use standard COCOeval.")

            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            cat_ids = list(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
            if self.per_class_AP:
                AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
                info += "per class AP:\n" + AP_table + "\n"
            if self.per_class_AR:
                AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
                info += "per class AR:\n" + AR_table + "\n"
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info

    def evaluate_prediction_solo(self, bbox_list, mask_list, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(bbox_list) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            if self.testdev:
                json.dump(bbox_list, open("./solo_bbox_testdev_2017.json", "w"))
                cocoDt_bbox = cocoGt.loadRes("./solo_bbox_testdev_2017.json")
                json.dump(mask_list, open("./solo_mask_testdev_2017.json", "w"))
                cocoDt_mask = cocoGt.loadRes("./solo_mask_testdev_2017.json")
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(bbox_list, open(tmp, "w"))
                cocoDt_bbox = cocoGt.loadRes(tmp)
                _, tmp2 = tempfile.mkstemp()
                json.dump(mask_list, open(tmp2, "w"))
                cocoDt_mask = cocoGt.loadRes(tmp2)
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval

                logger.warning("Use standard COCOeval.")

            cocoEval_bbox = COCOeval(cocoGt, cocoDt_bbox, annType[1])
            cocoEval_bbox.evaluate()
            cocoEval_bbox.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval_bbox.summarize()
            info += "bbox mAP:\n"
            info += redirect_string.getvalue()

            cocoEval_mask = COCOeval(cocoGt, cocoDt_mask, annType[0])
            cocoEval_mask.evaluate()
            cocoEval_mask.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval_mask.summarize()
            info += "mask mAP:\n"
            info += redirect_string.getvalue()

            return cocoEval_bbox.stats[0], cocoEval_bbox.stats[1], cocoEval_mask.stats[0], cocoEval_mask.stats[1], info
        else:
            return 0, 0, 0, 0, info
