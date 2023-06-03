#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

# add python path of this repo to sys.path
import sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from mmdet.core import launch
from mmdet.exp import get_exp
from mmdet.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger, load_ckpt


def make_parser():
    parser = argparse.ArgumentParser("MieMieDetection Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("-w", "--worker_num", type=int, default=1, help="worker num")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
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
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    configure_nccl()
    cudnn.benchmark = True

    rank = get_local_rank()

    file_name = os.path.join(exp.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    # 算法名字
    archi_name = exp.archi_name
    exp.data_num_workers = args.worker_num
    exp.eval_data_num_workers = args.worker_num

    # 新增算法时这里也要增加elif
    if archi_name == 'YOLOX':
        if args.conf is not None:
            exp.test_conf = args.conf
        if args.nms is not None:
            exp.nmsthre = args.nms
        if args.tsize is not None:
            exp.test_size = (args.tsize, args.tsize)
        model = exp.get_model()
        # logger.info("Model Summary: {}".format(get_model_info(archi_name, model, exp.test_size)))
        # logger.info("Model Structure:\n{}".format(str(model)))
        evaluator = exp.get_evaluator(args.batch_size, is_distributed, args.test, args.legacy)
    elif archi_name == 'PPYOLO':
        # PPYOLO使用的是matrix_nms，修改matrix_nms的配置。
        if args.conf is not None:
            if exp.nms_cfg['nms_type'] == 'matrix_nms':
                exp.nms_cfg['score_threshold'] = args.conf
                exp.nms_cfg['post_threshold'] = args.conf
            elif exp.nms_cfg['nms_type'] == 'multiclass_nms':
                exp.nms_cfg['score_threshold'] = args.conf
        if args.tsize is not None:
            exp.test_size = (args.tsize, args.tsize)
        model = exp.get_model()
        # logger.info("Model Summary: {}".format(get_model_info(archi_name, model, exp.test_size)))
        # logger.info("Model Structure:\n{}".format(str(model)))
        evaluator = exp.get_evaluator(args.batch_size, is_distributed, args.test)
    elif archi_name in ['PPYOLOE', 'PicoDet']:
        # PPYOLOE使用的是multiclass_nms，修改multiclass_nms的配置。
        if args.conf is not None:
            if exp.nms_cfg['nms_type'] == 'matrix_nms':
                exp.nms_cfg['score_threshold'] = args.conf
                exp.nms_cfg['post_threshold'] = args.conf
            elif exp.nms_cfg['nms_type'] == 'multiclass_nms':
                exp.nms_cfg['score_threshold'] = args.conf
        if args.tsize is not None:
            exp.test_size = [args.tsize, args.tsize]
            exp.head['eval_size'] = exp.test_size
        model = exp.get_model()
        # logger.info("Model Summary: {}".format(get_model_info(archi_name, model, exp.test_size)))
        # logger.info("Model Structure:\n{}".format(str(model)))
        evaluator = exp.get_evaluator(args.batch_size, is_distributed, args.test)
    elif archi_name in ['RTDETR', ]:
        if args.tsize is not None:
            exp.test_size = [args.tsize, args.tsize]
            exp.neck['eval_size'] = exp.test_size
            exp.transformer['eval_size'] = exp.test_size
        model = exp.get_model()
        # logger.info("Model Summary: {}".format(get_model_info(archi_name, model, exp.test_size)))
        # logger.info("Model Structure:\n{}".format(str(model)))
        evaluator = exp.get_evaluator(args.batch_size, is_distributed, args.test)
    else:
        raise NotImplementedError("Architectures \'{}\' is not implemented.".format(archi_name))


    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if not args.speed and not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
            if '/' not in ckpt_file and not os.path.exists(ckpt_file):
                ckpt_file = os.path.join(file_name, ckpt_file)
        logger.info("loading checkpoint from {}".format(ckpt_file))
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        model = load_ckpt(model, ckpt["model"])
        logger.info("loaded checkpoint done.")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert (
            not args.fuse and not is_distributed and args.batch_size == 1
        ), "TensorRT model is not support model fusing and distributed inferencing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    # start evaluate。新增算法时这里也要增加elif
    if archi_name == 'YOLOX':
        *_, summary = evaluator.evaluate_yolox(
            model, is_distributed, args.fp16, trt_file, decoder, exp.test_size
        )
    elif archi_name == 'PPYOLO':
        *_, summary = evaluator.evaluate_ppyolo(
            model, is_distributed, args.fp16, trt_file, exp.test_size
        )
    elif archi_name in ['PPYOLOE', 'PicoDet']:
        *_, summary = evaluator.evaluate_ppyoloe(
            model, is_distributed, args.fp16, trt_file, exp.test_size
        )
    elif archi_name == 'RTDETR':
        *_, summary = evaluator.evaluate_rtdetr(
            model, is_distributed, args.fp16, trt_file, exp.test_size
        )
    else:
        raise NotImplementedError("Architectures \'{}\' is not implemented.".format(archi_name))
    logger.info("\n" + summary)


if __name__ == "__main__":
    args = make_parser().parse_args()
    # 判断是否是调试状态
    isDebug = True if sys.gettrace() else False
    if isDebug:
        print('Debug Mode.')
        args.exp_file = '../' + args.exp_file
        args.ckpt = '../' + args.ckpt
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args, num_gpu),
    )
