#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: CAM-resnet.py

import cv2
import sys
import argparse
import numpy as np
import os
import multiprocessing


import tensorflow as tf
from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils import optimizer, gradproc
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.utils import viz

from utils import *
from utils_loc import *
from utils_args import *
from models_resnet import *

class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.uint8, [None,
                    args.final_size, args.final_size, 3], 'input'),
                tf.placeholder(tf.int32, [None], 'label'),
                tf.placeholder(tf.float32, [None, 2, 2], 'bbox')]

    def build_graph(self, image, label, bbox):
        ctx = get_current_tower_context()
        is_training = ctx.is_training

        image = image_preprocess(image, args, bgr=True)
        image = tf.transpose(image, [0, 3, 1, 2])
        label_onehot = tf.one_hot(label,args.classnum)
        image_summaries('input-images', image)

        logits, convmaps = resnet(image, DEPTH, args)

        # regularizer
        _, indices = tf.nn.top_k(logits, 5)
        indices = tf.identity(indices, name='top5')

        # Grad-CAM
        activation_map = tf.identity(convmaps, name='actmap')
        y_c = tf.reduce_sum(tf.multiply(logits, label_onehot), axis=1)
        target_conv_layer_grad = tf.identity(tf.gradients(y_c,
                                            convmaps)[0], name='grad')

        loss = compute_loss_and_error(logits, label)
        wd_cost = regularize_cost('.*/W',
                    l2_regularizer(1e-4), name='l2_regularize_loss')

        add_moving_summary(loss, wd_cost)
        return tf.add_n([loss, wd_cost], name='cost')

    def optimizer(self):
        lr = tf.get_variable('learning_rate',
                        initializer=args.base_lr, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
        add_moving_summary(lr)
        if args.load:
            gradprocs = [gradproc.ScaleGradient(
                [('conv0.*', 0.1), ('group[0-2].*', 0.1)])]
            return optimizer.apply_grad_processors(opt, gradprocs)
        else:
            return opt


if __name__ == '__main__':
    args = get_args()

    nr_gpu = get_nr_gpu()
    TOTAL_BATCH_SIZE = int(args.batch)
    BATCH_SIZE = TOTAL_BATCH_SIZE // nr_gpu
    args.batch = BATCH_SIZE # batch per gpu

    model = Model()
    DEPTH = args.depth

    log_dir = '/min-val-error-top1.index'

    if args.gradcam: gradcam(args, model, cam, log_dir)

    logdir = 'train_log/' + args.logdir

    logger.set_logger_dir(logdir)
    config = get_config(model, args)

    if args.load:
        if args.mode == 'se': args.load = \
                                    'pretrained/ImageNet-ResNet50-SE.npz'
        elif DEPTH == 34: args.load = 'pretrained/ImageNet-ResNet34.npz'
        elif DEPTH == 50: args.load = 'pretrained/ImageNet-ResNet50.npz'
        elif DEPTH == 101: args.load = 'pretrained/ImageNet-ResNet101.npz'
        config.session_init = get_model_loader(args.load)

    launch_train_with_config(config,
        SyncMultiGPUTrainerParameterServer(nr_gpu))

    bundle_test(args, model, cam, log_dir)
