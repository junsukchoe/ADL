# -*- coding: utf-8 -*-
# File: resnet_model.py

import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.tfutils.summary import *
from tensorpack.models import (
    Conv2D, MaxPooling, GlobalAvgPooling,
    BatchNorm, BNReLU, FullyConnected)
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from utils import image_summaries, att_summaries

def ADL(count, l, option):
    l = attention_drop(l, option)
    return l

def attention_drop(input_, option, is_training=None, flag=False):
    ctx = get_current_tower_context()
    is_training = ctx.is_training

    drop_prob = 1-option.keep_prob
    drop_thr = option.threshold

    if is_training: # ADL is only applied during training phase
        shape = input_.get_shape().as_list() # [batch, channel, height, width]

        # Generate self-attention map
        attention = tf.reduce_mean(input_, axis=1, keepdims=True) # [batch, 1, height, width]

        # Generate importance map
        importance_map = tf.sigmoid(attention)

        # Generate drop mask
        max_val = tf.reduce_max(attention, axis=[1,2,3], keepdims=True)
        thr_val = max_val * drop_thr
        drop_mask = tf.cast(attention < thr_val, dtype=tf.float32, name='drop_mask')

        # Random selection
        random_tensor = tf.random_uniform([], drop_prob, 1.+drop_prob)
        binary_tensor = tf.cast(tf.floor(random_tensor), dtype=tf.float32)
        selected_map = (1. - binary_tensor) * importance_map + binary_tensor * drop_mask

        # Spatialwise multiplication to input feature map
        output = input_ * selected_map
        return output

    else: # during testing phase
        return input_

def convnormrelu(x, name, chan, option, kernel_size=3, padding='SAME'):
    x = Conv2D(name, x, chan, kernel_size=kernel_size, padding=padding)
    x = tf.nn.relu(x, name=name + '_relu')

    return x
