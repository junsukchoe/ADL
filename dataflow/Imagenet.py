import numpy as np
import cv2

from os.path import join as ospj
from tensorpack.dataflow.base import RNGDataFlow

__all__ = ['ImagenetMeta', 'Imagenet']


class ImagenetMeta(object):
    def __init__(self):
        self.meta_dir = ospj('labels', 'ILSVRC')

    def get_synset_words_1000(self):
        fname = ospj(self.meta_dir, 'words.txt')
        lines = [x.strip().split(' ', 1) for x in open(fname).readlines()]
        return dict(lines)

    def get_synset_1000(self):
        fname = ospj(self.meta_dir, 'wnids.txt')
        lines = [x.strip() for x in open(fname).readlines()]
        return dict(enumerate(lines))

    def get_image_list(self, split):
        fname = ospj(self.meta_dir, '{}.txt'.format(split))
        with open(fname) as f:
            ret = []
            for line in f.readlines():
                if split == 'train':
                    name, cls = line.strip().split()
                    xa = ya = xb = yb = 1
                elif split == 'val':
                    name, cls, xa, ya, xb, yb = line.strip().split()
                else:
                    raise KeyError("Unavailable split: {}".format(split))

                bbox = np.array(
                    [(float(xa), float(ya)), (float(xb), float(yb))],
                    dtype=np.float)
                ret.append((name.strip(), int(cls), bbox))
        return ret


class ImagenetFiles(RNGDataFlow):
    def __init__(self, data_dir, split, shuffle=None):
        if shuffle is None:
            shuffle = split == 'train'
        meta = ImagenetMeta()

        self.data_dir = ospj(data_dir, split)
        self.shuffle = shuffle
        self.imglist = meta.get_image_list(split)

    def size(self):
        return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            fname, label, bbox = self.imglist[k]
            fname = ospj(self.data_dir, fname)
            yield [fname, label, bbox]


class Imagenet(ImagenetFiles):
    def __init__(self, data_dir, split, shuffle=None):
        super(Imagenet, self).__init__(data_dir, split, shuffle)

    def get_data(self):
        for fname, label, bbox in super(Imagenet, self).get_data():
            img = cv2.imread(fname, cv2.IMREAD_COLOR)
            yield [img, label, bbox]
