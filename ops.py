import tensorflow as tf
import numpy as np

from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.models import Conv2D


def gating_op(input_, option):
    if option.method_name == 'CAM':
        output = input_
    elif option.method_name == 'ADL':
        output = attention_based_dropout(input_, option)
    else:
        raise KeyError("Unavailable method: {}".format(option.method_name))

    return output


def attention_based_dropout(input_, option):
    def _get_importance_map(attention):
        return tf.sigmoid(attention)

    def _get_drop_mask(attention, drop_thr):
        max_val = tf.reduce_max(attention, axis=[1, 2, 3], keepdims=True)
        thr_val = max_val * drop_thr
        return tf.cast(attention < thr_val, dtype=tf.float32, name='drop_mask')

    def _select_component(importance_map, drop_mask, drop_prob):
        random_tensor = tf.random_uniform([], drop_prob, 1. + drop_prob)
        binary_tensor = tf.cast(tf.floor(random_tensor), dtype=tf.float32)
        return (1. - binary_tensor) * importance_map + binary_tensor * drop_mask

    ctx = get_current_tower_context()
    is_training = ctx.is_training

    drop_prob = 1 - option.adl_keep_prob
    drop_thr = option.adl_threshold

    if is_training:
        attention_map = tf.reduce_mean(input_, axis=1, keepdims=True)
        importance_map = _get_importance_map(attention_map)
        drop_mask = _get_drop_mask(attention_map, drop_thr)
        selected_map = _select_component(importance_map, drop_mask, drop_prob)
        output = input_ * selected_map
        return output

    else:
        return input_


def convnormrelu(x, name, chan, kernel_size=3, padding='SAME'):
    x = Conv2D(name, x, chan, kernel_size=kernel_size, padding=padding)
    x = tf.nn.relu(x, name=name + '_relu')
    return x
