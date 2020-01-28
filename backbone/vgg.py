import tensorflow as tf

from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.models import Conv2D
from tensorpack.models import MaxPooling
from tensorpack.models import GlobalAvgPooling
from tensorpack.models import BatchNorm
from tensorpack.models import FullyConnected

from ops import gating_op
from ops import convnormrelu

__all__ = ['vgg_gap']


@auto_reuse_variable_scope
def vgg_gap(image, option):
    with argscope(Conv2D, use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(scale=2.)), \
         argscope([Conv2D, MaxPooling, BatchNorm, GlobalAvgPooling],
                  data_format='channels_first'):

        l = convnormrelu(image, 'conv1_1', 64)
        if option.gating_position[11]: l = gating_op(l, option)
        l = convnormrelu(l, 'conv1_2', 64)
        if option.gating_position[12]: l = gating_op(l, option)
        l = MaxPooling('pool1', l, 2)
        if option.gating_position[1]: l = gating_op(l, option)

        l = convnormrelu(l, 'conv2_1', 128)
        if option.gating_position[21]: l = gating_op(l, option)
        l = convnormrelu(l, 'conv2_2', 128)
        if option.gating_position[22]: l = gating_op(l, option)
        l = MaxPooling('pool2', l, 2)
        if option.gating_position[2]: l = gating_op(l, option)

        l = convnormrelu(l, 'conv3_1', 256)
        if option.gating_position[31]: l = gating_op(l, option)
        l = convnormrelu(l, 'conv3_2', 256)
        if option.gating_position[32]: l = gating_op(l, option)
        l = convnormrelu(l, 'conv3_3', 256)
        if option.gating_position[33]: l = gating_op(l, option)
        l = MaxPooling('pool3', l, 2)
        if option.gating_position[3]: l = gating_op(l, option)

        l = convnormrelu(l, 'conv4_1', 512)
        if option.gating_position[41]: l = gating_op(l, option)
        l = convnormrelu(l, 'conv4_2', 512)
        if option.gating_position[42]: l = gating_op(l, option)
        l = convnormrelu(l, 'conv4_3', 512)
        if option.gating_position[43]: l = gating_op(l, option)
        l = MaxPooling('pool4', l, 2)
        if option.gating_position[4]: l = gating_op(l, option)

        l = convnormrelu(l, 'conv5_1', 512)
        if option.gating_position[51]: l = gating_op(l, option)
        l = convnormrelu(l, 'conv5_2', 512)
        if option.gating_position[52]: l = gating_op(l, option)
        l = convnormrelu(l, 'conv5_3', 512)
        if option.gating_position[53]: l = gating_op(l, option)

        convmaps = convnormrelu(l, 'new', 1024)
        if option.gating_position[6]: convmaps = gating_op(l, option)

        p_logits = GlobalAvgPooling('gap', convmaps)
        logits = FullyConnected('linear',
                                p_logits, option.number_of_class,
                                kernel_initializer=tf.random_normal_initializer(
                                    stddev=0.01))

    return logits, convmaps
