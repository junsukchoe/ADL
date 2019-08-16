#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: utils.py

# Written by Junsuk Choe <skykite@yonsei.ac.kr>
# Utility code for Wearkly-Supervised Object Localization (WSOL).


import cv2
import numpy as np
import multiprocessing
import tensorflow as tf
import random
import os
import sys

from abc import abstractmethod

from tensorpack import *
from tensorpack.dataflow import (
    AugmentImageComponent, AugmentImageCoordinates, PrefetchDataZMQ,
    BatchData, MultiThreadMapData, LMDBSerializer)
from tensorpack.predict import PredictConfig, SimpleDatasetPredictor
from tensorpack.utils.stats import RatioCounter
from tensorpack.models import regularize_cost
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils import logger
from dataflow import Imagenet, CUB200
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.tfutils import argscope, get_model_loader, model_utils

def gradcam(option, model, cam, log_dir):
    option.batch = int(100)
    option.load = 'train_log/' + option.logdir + log_dir
    cam(model, args, gradcam=True)
    sys.exit()

def bundle_test(option, model, cam, log_dir):
    option.batch = int(100)
    option.camrelu = False
    for i in range(7):
        option.locthr= 0.05 + i * 0.05
        option.load = 'train_log/' + option.logdir + log_dir
        cam(model, option, gradcam=True)

def image_summaries(name, img, max_outputs=1):
    img = tf.transpose(img, [0, 2, 3, 1])
    chan = tf.unstack(img,axis=-1)
    img_restack = tf.stack([chan[2],chan[1],chan[0]],axis=-1,name=name)
    tf.summary.image(name, img_restack, max_outputs=max_outputs)

def att_summaries(name, img, att, option, flag=False):
    if option.chlast: att = tf.transpose(att, [0,3,1,2])

    att = tf.transpose(att, [0,2,3,1])
    att = tf.image.resize_images(att, [option.final_size,
                                    option.final_size], align_corners=True)
    att = tf.transpose(att, [0,3,1,2])

    temp = tf.concat([att, att, att], axis=1)
    image_summaries(name, temp)

def get_data(train_or_test, option):
    isTrain = train_or_test == 'train'
    parallel = multiprocessing.cpu_count() // 2
    datadir = option.data
    if option.imagenet:
        ds = Imagenet.Imagenet(
            datadir, train_or_test, option.dataname, shuffle=isTrain)
    elif option.cub:
        ds = CUB200.CUB200(
            datadir, train_or_test, option.dataname, shuffle=isTrain)
    augmentors = fbresnet_augmentor(isTrain, option=option)
    print(augmentors)
    ds = AugmentImageCoordinates(ds, augmentors, coords_index=2,
                                                            copy=False)
    if isTrain: ds = PrefetchDataZMQ(ds, parallel)
    ds = BatchData(ds, int(option.batch), remainder=not isTrain)
    return ds

def get_config(model, option):
    dataset_train = get_data('train', option)
    dataset_val = get_data('val', option)

    nr_tower = max(get_nr_gpu(), 1)
    total_batch = int(option.batch) * nr_tower
    lr_string = 'learning_rate'
    infs = [ClassificationError('wrong-top1', 'val-error-top1'),
            ClassificationError('wrong-top5', 'val-error-top5')]

    START_LR = option.base_lr
    BASE_LR = START_LR

    callbacks = [
            ModelSaver(max_to_keep=1, keep_checkpoint_every_n_hours=1000),
            EstimatedTimeLeft(),
            MinSaver('val-error-top1'),
            ScheduledHyperParamSetter(lr_string,
                                    [(0, min(START_LR, BASE_LR)),
                                    (30, BASE_LR * 1e-1),
                                    (60, BASE_LR * 1e-2),
                                    (90, BASE_LR * 1e-3),
                                    (100, BASE_LR * 1e-4)]),
        ]

    if nr_tower == 1:
        call = [PeriodicTrigger(InferenceRunner(dataset_val,
                infs), every_k_epochs=2)]
    else:
        call = [PeriodicTrigger(DataParallelInferenceRunner(dataset_val,
                infs, list(range(nr_tower))), every_k_epochs=2)]

    call.extend(callbacks)

    input = QueueInput(dataset_train)
    input = StagingInput(input, nr_stage=1)

    if option.cub:
        steps_per_epoch = 25 * (256 / total_batch) * option.stepscale
    else:
        steps_per_epoch = 5000 * (256 / total_batch) * option.stepscale

    return TrainConfig(
        model=model,
        data=input,
        callbacks=call,
        steps_per_epoch=int(steps_per_epoch),
        max_epoch=option.epoch,
    )


def fbresnet_augmentor(isTrain, option):
    """
    Augmentor used in fb.resnet.torch, for BGR images in range [0,255].
    """
    if isTrain:
        augmentors = [
                    imgaug.ToFloat32(),
                    imgaug.Resize((option.final_size+32,
                                        option.final_size+32)),
                    imgaug.RandomCrop((option.final_size,
                                        option.final_size))]


        flip = [imgaug.Flip(horiz=True), imgaug.ToUint8()]
        augmentors.extend(flip)

    else: # testing
        augmentors = [
                imgaug.ToFloat32(),
                imgaug.Resize((option.final_size+32, option.final_size+32)),
                imgaug.CenterCrop((option.final_size, option.final_size)),
                imgaug.ToUint8()]

    return augmentors


def image_preprocess(image, option, bgr=True):
    with tf.name_scope('image_preprocess'):
        if image.dtype.base_dtype != tf.float32:
            image = tf.cast(image, tf.float32)

        image = image * (1.0 / 255)
        mean = [0.485, 0.456, 0.406]    # rgb
        std = [0.229, 0.224, 0.225]
        if bgr:
            mean = mean[::-1]
            std = std[::-1]
        image_mean = tf.constant(mean, dtype=tf.float32)
        image_std = tf.constant(std, dtype=tf.float32)
        image = (image - image_mean) / image_std

        return image

def compute_loss_and_error(logits, label):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(\
                                    logits=logits, labels=label)

    loss = tf.reduce_mean(loss, name='xentropy-loss')

    def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
            with tf.name_scope('prediction_incorrect'):
                x = tf.logical_not(tf.nn.in_top_k(logits, label, topk))
            return tf.cast(x, tf.float32, name=name)

    wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
    add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

    wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
    add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))

    return loss

def compute_att_error(logits, label):
    def prediction_incorrect(logits,
                        label, topk=1, name='incorrect_vector_att'):
        with tf.name_scope('prediction_incorrect'):
            x = tf.logical_not(tf.nn.in_top_k(logits, label, topk))
        return tf.cast(x, tf.float32, name=name)

    wrong = prediction_incorrect(logits, label, 1, name='wrong-top1_att')
    add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1_att'))

    wrong = prediction_incorrect(logits, label, 5, name='wrong-top5_att')
    add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5_att'))
