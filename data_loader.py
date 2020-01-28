import multiprocessing

from tensorpack import imgaug
from tensorpack.dataflow import AugmentImageCoordinates
from tensorpack.dataflow import PrefetchDataZMQ
from tensorpack.dataflow import BatchData

import dataflow


def fbresnet_augmentor(is_training, option):
    if is_training:
        augmentors = [
            imgaug.ToFloat32(),
            imgaug.Resize((option.final_size + 32,
                           option.final_size + 32)),
            imgaug.RandomCrop((option.final_size,
                               option.final_size))]

        flip = [imgaug.Flip(horiz=True), imgaug.ToUint8()]
        augmentors.extend(flip)

    else:
        augmentors = [
            imgaug.ToFloat32(),
            imgaug.Resize((option.final_size + 32, option.final_size + 32)),
            imgaug.CenterCrop((option.final_size, option.final_size)),
            imgaug.ToUint8()]

    return augmentors


def get_data_flow(split, is_training, option):
    if option.dataset_name == 'ILSVRC':
        ds = dataflow.Imagenet(option.data_dir, split, shuffle=is_training)
    elif option.dataset_name == 'CUB':
        ds = dataflow.CUB(option.data_dir, split, shuffle=is_training)
    else:
        raise KeyError("Unavailable dataset: {}".format(option.dataset_name))
    return ds


def get_data(split, option):
    is_training = split == 'train'
    parallel = multiprocessing.cpu_count() // 2
    ds = get_data_flow(split, is_training, option)
    augmentors = fbresnet_augmentor(is_training, option)
    ds = AugmentImageCoordinates(ds, augmentors, coords_index=2, copy=False)
    if is_training:
        ds = PrefetchDataZMQ(ds, parallel)
    ds = BatchData(ds, option.batch_size, remainder=not is_training)
    return ds
