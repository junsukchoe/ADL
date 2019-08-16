# -*- coding: utf-8 -*-
# File: resnet_model.py

import tensorflow as tf

from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.models import (
    Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm, BNReLU, FullyConnected)

from ops import *
from utils import image_summaries, att_summaries


def resnet(input_, DEPTH, option):
    ctx = get_current_tower_context()
    is_training = ctx.is_training

    mode = option.mode
    basicblock = preresnet_basicblock \
                    if mode == 'preact' else resnet_basicblock
    bottleneck = {
        'resnet': resnet_bottleneck,
        'preact': preresnet_bottleneck,
        'se': se_resnet_bottleneck}[mode]

    cfg = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck),
            152: ([3, 8, 36, 3], bottleneck)
        }
    defs, block_func = cfg[DEPTH]
    group_func = preresnet_group if mode == 'preact' else resnet_group

    with argscope(Conv2D, use_bias=False, kernel_initializer= \
            tf.variance_scaling_initializer(scale=2.0, mode='fan_out')), \
            argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm],
                                            data_format='channels_first'):

        l = Conv2D('conv0', input_, 64, 7, strides=2, activation=BNReLU) # 112
        if option.attdrop[0]: l = ADL(0, l, option)

        l = MaxPooling('pool0', l, 3, strides=2, padding='SAME') # 56
        if option.attdrop[1]: l = ADL(1, l, option)

        l = group_func('group0', l, block_func, 64, defs[0], 1, option) # 56
        if option.attdrop[2]: l = ADL(2, l, option)

        l = group_func('group1', l, block_func, 128, defs[1], 2, option) # 28
        if option.attdrop[3]: l = ADL(3, l, option)

        l = group_func('group2', l, block_func, 256, defs[2], 2, option) # 14
        if option.attdrop[4]: l = ADL(4, l, option)

        l = group_func('group3', l, block_func, 512, defs[3],
                                              option.laststride, option) # 7
        if option.attdrop[5]: l = ADL(5, l, option)

        prelogits = GlobalAvgPooling('gap', l)
        logits = FullyConnected('linearnew', prelogits, option.classnum)

    return logits, l

def resnet_group(name, l, block_func, features, count, stride, option):
    if features == 64: k = 10
    elif features == 128: k = 20
    elif features == 256: k = 30
    elif features == 512: k = 40
    # 31 41 51 5 or 31 41 5?
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                k = k + 1
                l = block_func(
                    option, l, features, stride \
                                if i == 0 else 1, count=k)

    return l

def preresnet_group(name, l, block_func, features, count, stride, option):
    if features == 64: k = 10
    elif features == 128: k = 20
    elif features == 256: k = 30
    elif features == 512: k = 40

    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                # first block doesn't need activation
                l = block_func(option, l, features,
                               stride if i == 0 else 1,
                               'no_preact' if i == 0 else 'bnrelu', count=k)
        # end of each group need an extra activation
        l = BNReLU('bnlast', l)
    return l

def resnet_shortcut(l, n_out, stride, activation=tf.identity):
    data_format = get_arg_scope()['Conv2D']['data_format']
    n_in = l.get_shape().as_list()[1 \
                    if data_format in ['NCHW', 'channels_first'] else 3]
    if n_in != n_out:   # change dimension when channel is not the same
        return Conv2D('convshortcut',
                    l, n_out, 1, strides=stride, activation=activation)
    else:
        return l


def apply_preactivation(l, preact):
    if preact == 'bnrelu':
        shortcut = l    # preserve identity mapping
        l = BNReLU('preact', l)
    else:
        shortcut = l
    return l, shortcut


def get_bn(option, zero_init=False):
    """
    Zero init gamma is good for resnet. See https://arxiv.org/abs/1706.02677.
    """
    if zero_init:
        return lambda x, name=None: BatchNorm('bn', x,
                gamma_initializer=tf.zeros_initializer())
    else:
        return lambda x, name=None: BatchNorm('bn', x)


def preresnet_basicblock(option,
                l, ch_out, stride, preact, ADL_Flag=False, count=None):
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3)
    return l + resnet_shortcut(shortcut, ch_out, stride)


def preresnet_bottleneck(option,
                l, ch_out, stride, preact, ADL_Flag=False, count=None):
    # stride is applied on the second conv, following fb.resnet.torch
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1)

    out = l + resnet_shortcut(shortcut, ch_out * 4, stride)
    if option.attdrop[count]: out = ADL(count, out, option)
    return out



def resnet_basicblock(option, l, ch_out, stride, ADL_Flag=False, count=None):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3,
                    activation=get_bn(option, zero_init=True))

    out = l + resnet_shortcut(shortcut,
                    ch_out, stride,
                    activation=get_bn(option, zero_init=False))

    out = tf.nn.relu(out)
    if option.attdrop[count]: out = ADL(count, out, option)
    return out


def resnet_bottleneck(option, l, ch_out,
                    stride, stride_first=False, ADL_Flag=False, count=None):
    """
    stride_first: original resnet put stride on first conv.
    fb.resnet.torch put stride on second conv.
    """
    shortcut = l
    l = Conv2D('conv1',
        l, ch_out, 1, strides=stride \
                if stride_first else 1, activation=BNReLU)
    l = Conv2D('conv2',
        l, ch_out, 3, strides=1 \
                if stride_first else stride, activation=BNReLU)
    l = Conv2D('conv3',
        l, ch_out * 4, 1, activation=get_bn(option, zero_init=True))
    out = l + resnet_shortcut(shortcut,
        ch_out * 4, stride, activation=get_bn(option, zero_init=False))

    out = tf.nn.relu(out)
    if option.attdrop[count]: out = ADL(count, out, option)
    return out

def se_resnet_bottleneck(option,
                        l, ch_out, stride, ADL_Flag=False, count=None):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1,
                                activation=get_bn(option, zero_init=True))

    squeeze = GlobalAvgPooling('gap', l)
    squeeze = FullyConnected('fc1',
                                squeeze, ch_out // 4, activation=tf.nn.relu)
    squeeze = FullyConnected('fc2',
                                squeeze, ch_out * 4, activation=tf.nn.sigmoid)
    data_format = get_arg_scope()['Conv2D']['data_format']
    ch_ax = 1 if data_format in ['NCHW', 'channels_first'] else 3
    shape = [-1, 1, 1, 1]
    shape[ch_ax] = ch_out * 4
    l = l * tf.reshape(squeeze, shape)

    out = l + resnet_shortcut(shortcut,
                ch_out * 4, stride,
                activation=get_bn(option, zero_init=False))

    out = tf.nn.relu(out)
    if option.attdrop[count]: out = ADL(count, out, option)
    return out
