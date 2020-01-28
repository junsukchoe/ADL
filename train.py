import tensorflow as tf
import sys

from tensorpack import ModelSaver
from tensorpack import ClassificationError
from tensorpack import EstimatedTimeLeft
from tensorpack import MinSaver
from tensorpack import ScheduledHyperParamSetter
from tensorpack import QueueInput
from tensorpack import StagingInput
from tensorpack import InferenceRunner
from tensorpack.models.regularize import regularize_cost
from tensorpack.models.regularize import l2_regularizer
from tensorpack.tfutils import optimizer
from tensorpack.tfutils import gradproc
from tensorpack.tfutils import get_model_loader
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.train.config import TrainConfig
from tensorpack.train.interface import launch_train_with_config
from tensorpack.train.model_desc import ModelDesc
from tensorpack.train.trainers import SyncMultiGPUTrainerParameterServer
from tensorpack.utils import logger
from tensorpack.utils.gpu import get_nr_gpu

from config import get_args
from data_loader import get_data
from evaluate import evaluate_wsol
from os.path import join as ospj
from util import image_summaries

import backbone as arch

_CKPT_NAMES = {
    'resnet50_se': 'pretrained/ResNet50-SE.npz',
    'vgg_gap': 'pretrained/vgg16.npz',
}

_WEIGHT_DECAY = {
    'resnet50_se': 1e-4,
    'vgg_gap': 5e-4,
}

_LR_SCALE = {
    'resnet50_se': [('conv0.*', 0.1), ('group[0-2].*', 0.1)],
    'vgg_gap': [('conv.*', 0.1), ('fc.*', 0.1)],
}


class Model(ModelDesc):
    def __init__(self, args):
        self.args = args

    def inputs(self):
        input_shape = [None, self.args.final_size, self.args.final_size, 3]
        return [tf.placeholder(tf.uint8, input_shape, 'input'),
                tf.placeholder(tf.int32, [None], 'label'),
                tf.placeholder(tf.float32, [None, 2, 2], 'bbox')]

    def build_graph(self, image, label, bbox):
        image, label_onehot = self._pre_process_inputs(image, label)
        logit, convmap = arch.__dict__[self.args.arch_name](image, self.args)
        self._prepare_cam(logit, convmap, label_onehot)
        loss = self._get_loss(logit, label)
        return loss

    def optimizer(self):
        lr = tf.get_variable('learning_rate',
                             initializer=self.args.base_lr, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
        add_moving_summary(lr)
        if self.args.use_pretrained_model:
            gradprocs = [gradproc.ScaleGradient(_LR_SCALE[self.args.arch_name])]
            return optimizer.apply_grad_processors(opt, gradprocs)
        else:
            return opt

    def _pre_process_inputs(self, image, label):
        image = self._image_preprocess(image, is_bgr=True)
        image = tf.transpose(image, [0, 3, 1, 2])
        label_onehot = tf.one_hot(label, self.args.number_of_class)
        image_summaries('input-images', image)
        return image, label_onehot

    def _image_preprocess(self, image, is_bgr=True):
        with tf.name_scope('image_preprocess'):
            if image.dtype.base_dtype != tf.float32:
                image = tf.cast(image, tf.float32)

            image = image * (1.0 / 255)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            if is_bgr:
                mean = mean[::-1]
                std = std[::-1]
            image_mean = tf.constant(mean, dtype=tf.float32)
            image_std = tf.constant(std, dtype=tf.float32)
            image = (image - image_mean) / image_std

            return image

    def _prepare_cam(self, logits, convmaps, label_onehot):
        _, indices = tf.nn.top_k(logits, 5)
        tf.identity(indices, name='top5')
        tf.identity(convmaps, name='actmap')
        y_c = tf.reduce_sum(tf.multiply(logits, label_onehot), axis=1)
        tf.identity(tf.gradients(y_c, convmaps)[0], name='grad')

    def _get_loss(self, logits, label):
        loss = self._compute_loss_and_error(logits, label)
        wd_cost = regularize_cost('.*/W', l2_regularizer(
            _WEIGHT_DECAY[self.args.arch_name]), name='l2_regularize_loss')
        add_moving_summary(loss, wd_cost)

        return tf.add_n([loss, wd_cost], name='cost')

    def _compute_loss_and_error(self, logits, label):
        def prediction_incorrect(logits_, label_, topk=1,
                                 name='incorrect_vector'):
            with tf.name_scope('prediction_incorrect'):
                x = tf.logical_not(tf.nn.in_top_k(logits_, label_, topk))
            return tf.cast(x, tf.float32, name=name)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                              labels=label)
        loss = tf.reduce_mean(loss, name='xentropy-loss')

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))

        return loss


def get_callbacks(dataset_val, option):
    cls_metric = [ClassificationError('wrong-top1', 'val-error-top1'),
                  ClassificationError('wrong-top5', 'val-error-top5')]

    callbacks = [
        ModelSaver(max_to_keep=1, keep_checkpoint_every_n_hours=1000),
        EstimatedTimeLeft(),
        InferenceRunner(dataset_val, cls_metric),
        MinSaver('val-error-top1'),
        ScheduledHyperParamSetter('learning_rate',
                                  [(0, option.base_lr),
                                   (30, option.base_lr * 1e-1),
                                   (60, option.base_lr * 1e-2),
                                   (90, option.base_lr * 1e-3),
                                   (100, option.base_lr * 1e-4)]),
    ]
    return callbacks


def get_steps_per_epoch(option):
    nr_gpu = get_nr_gpu()
    total_batch = option.batch_size * nr_gpu

    if option.dataset_name == 'CUB':
        steps_per_epoch = 25 * (256 / total_batch) * option.stepscale
    elif option.dataset_name == 'ILSVRC':
        steps_per_epoch = 5000 * (256 / total_batch) * option.stepscale
    else:
        raise KeyError("Unavailable dataset: {}".format(option.dataset_name))
    return int(steps_per_epoch)


def get_config(model, option):
    dataset_train = get_data('train', option)
    dataset_val = get_data('val', option)
    callbacks = get_callbacks(dataset_val, option)
    steps_per_epoch = get_steps_per_epoch(option)

    return TrainConfig(
        model=model,
        data=StagingInput(QueueInput(dataset_train), nr_stage=1),
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        max_epoch=option.epoch,
    )


def main():
    args = get_args()
    nr_gpu = get_nr_gpu()
    args.batch_size = args.batch_size // nr_gpu

    model = Model(args)

    if args.evaluate:
        evaluate_wsol(args, model, interval=False)
        sys.exit()

    logger.set_logger_dir(ospj('train_log', args.log_dir))
    config = get_config(model, args)

    if args.use_pretrained_model:
        config.session_init = get_model_loader(_CKPT_NAMES[args.arch_name])

    launch_train_with_config(config,
                             SyncMultiGPUTrainerParameterServer(nr_gpu))

    evaluate_wsol(args, model, interval=True)


if __name__ == '__main__':
    main()
