#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: utils_args.py

# Written by Junsuk Choe <skykite@yonsei.ac.kr>
# Parsing arguments code for Wearkly-Supervised Object Localization (WSOL).


import cv2
import numpy as np
import multiprocessing
import tensorflow as tf
from abc import abstractmethod
import argparse
import os

from tensorpack import imgaug, dataset, ModelDesc
from tensorpack.dataflow import (
    AugmentImageComponent, PrefetchDataZMQ,
    BatchData, MultiThreadMapData)
from tensorpack.predict import PredictConfig, SimpleDatasetPredictor
from tensorpack.utils.stats import RatioCounter
from tensorpack.models import regularize_cost
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils import logger
from random import choice, randrange

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def to_bool(list, totalnum):
    bool_list = [False] * totalnum
    for i in list:
        bool_list[i] = True
    return bool_list

def get_args():
    parser = argparse.ArgumentParser()

    # Common - Required
    parser.add_argument('--gpu',
                        help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--logdir', help='log directory name')

    # Common - Default
    parser.add_argument('--epoch', help='max epoch', type=int, default=105)
    parser.add_argument('--dataname', help='dataset name', default='all')
    parser.add_argument('--steps',
                help='steps_per_epoch', default=5000, type=int)
    parser.add_argument('--final-size', type=int, default=224)
    parser.add_argument('--chlast', action='store_true')

    # dataset
    parser.add_argument('--cub', action='store_true')
    parser.add_argument('--imagenet', action='store_true')
    parser.add_argument('--classnum', type=int, default=1000)

    # Training
    parser.add_argument('--batch', default=256)
    parser.add_argument('--base-lr', type=float, default=0.01)
    parser.add_argument('--stepscale', type=float, default=1.)
    parser.add_argument('--finetune', action='store_true')

    # resnet
    parser.add_argument('--depth', type=int, default=50)
    parser.add_argument('--mode', help='resnet type', default='resnet')
    parser.add_argument('--laststride', type=int, default=1)

    # drop_block
    parser.add_argument("--interp_init", type=float, default=1.0)
    parser.add_argument("--interp_epoch", type=int, default=100)

    # ADL
    parser.add_argument('--attdrop', nargs='+', type=int)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--keep_prob', type=float, default=0.25)

    # Test
    parser.add_argument('--val', type=int, default=1)
    parser.add_argument('--load', help='load model')
    parser.add_argument('--gradcam', action='store_true')
    parser.add_argument('--camrelu', action='store_true')
    parser.add_argument('--locthr', type=float, default=0.2)

    global args
    args = parser.parse_args()
    dropcount = 100
    if args.attdrop is not None: args.attdrop = to_bool(args.attdrop, dropcount)
    else: args.attdrop = [False]*dropcount

    if args.gpu: os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.cub:
        args.laststride = 1
        args.stepscale = 5.0
        args.epoch = 105
    elif args.imagenet:
        args.laststride = 1
        args.stepscale = 0.2
        args.epoch = 105

    if args.imagenet: args.classnum = 1000
    elif args.cub: args.classnum = 200

    return args
