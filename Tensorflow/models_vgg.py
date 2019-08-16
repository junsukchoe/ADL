# -*- coding: utf-8 -*-
# File: vgg_model.py

import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.tfutils.summary import *
from tensorpack.models import (
    Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm, BNReLU, FullyConnected)

from tensorpack.tfutils.tower import get_current_tower_context
from ops import *
from utils import image_summaries, att_summaries

@auto_reuse_variable_scope
def vgg_gap(image, option, importance=False):
    ctx = get_current_tower_context()
    is_training = ctx.is_training

    with argscope(Conv2D,
        kernel_initializer=tf.variance_scaling_initializer(scale=2.)), \
            argscope([Conv2D, MaxPooling, BatchNorm, GlobalAvgPooling],
                data_format='channels_first'):

        l = convnormrelu(image, 'conv1_1', 64, option)
        if option.attdrop[11]: l = ADL(11, l, option)
        l = convnormrelu(l, 'conv1_2', 64, option)
        if option.attdrop[12]: l = ADL(12, l, option)
        l = MaxPooling('pool1', l, 2)
        if option.attdrop[1]:  l = ADL(1,  l, option)

        l = convnormrelu(l, 'conv2_1', 128, option)
        if option.attdrop[21]: l = ADL(21, l, option)
        l = convnormrelu(l, 'conv2_2', 128, option)
        if option.attdrop[22]: l = ADL(21, l, option)
        l = MaxPooling('pool2', l, 2)
        if option.attdrop[2]:  l = ADL(2,  l, option)

        l = convnormrelu(l, 'conv3_1', 256, option)
        if option.attdrop[31]: l = ADL(31, l, option)
        l = convnormrelu(l, 'conv3_2', 256, option)
        if option.attdrop[32]: l = ADL(32, l, option)
        l = convnormrelu(l, 'conv3_3', 256, option)
        if option.attdrop[33]: l = ADL(33, l, option)
        l = MaxPooling('pool3', l, 2)
        if option.attdrop[3]:  l = ADL(3,  l, option)

        l = convnormrelu(l, 'conv4_1', 512, option)
        if option.attdrop[41]: l = ADL(41, l, option)
        l = convnormrelu(l, 'conv4_2', 512, option)
        if option.attdrop[42]: l = ADL(42, l, option)
        l = convnormrelu(l, 'conv4_3', 512, option)
        if option.attdrop[43]: l = ADL(43, l, option)
        l = MaxPooling('pool4', l, 2)
        if option.attdrop[4]:  l = ADL(4,  l, option)

        l = convnormrelu(l, 'conv5_1', 512, option)
        if option.attdrop[51]: l = ADL(51, l, option)
        l = convnormrelu(l, 'conv5_2', 512, option)
        if option.attdrop[52]: l = ADL(52, l, option)
        l = convnormrelu(l, 'conv5_3', 512, option)
        if option.attdrop[53]: l = ADL(53, l, option)

        convmaps = convnormrelu(l, 'new', 1024, option)
        if option.attdrop[6]:  l = ADL(6,  l, option)

        pre_logits = GlobalAvgPooling('gap', convmaps)
        logits = FullyConnected('linear',
            pre_logits, option.classnum,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01))

        return logits, convmaps
