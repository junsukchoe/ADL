import numpy as np
import cv2

from os.path import join as ospj
from tensorpack.dataflow.base import RNGDataFlow

__all__ = ['CUB', 'CUBMeta']


class CUBMeta(object):
    def __init__(self):
        self.meta_dir = ospj('labels', 'CUB')

    def get_synset_words_1000(self):
        fname = ospj(self.meta_dir, 'words.txt')
        lines = [x.strip().split(' ', 1) for x in open(fname).readlines()]
        return dict(lines)

    def get_synset_1000(self):
        fname = ospj(self.meta_dir, 'wnids.txt')
        lines = [x.strip() for x in open(fname).readlines()]
        return dict(enumerate(lines))

    def get_image_list(self, name):
        fname = ospj(self.meta_dir, '{}.txt'.format(name))
        with open(fname) as f:
            ret = []
            for line in f.readlines():
                name, cls, xa, ya, xb, yb = line.strip().split()
                bbox = np.array(
                    [(float(xa), float(ya)), (float(xb), float(yb))],
                    dtype=np.float)
                ret.append((name.strip(), int(cls), bbox))
        return ret


class CUBFiles(RNGDataFlow):
    def __init__(self, data_dir, split, shuffle=None):
        if shuffle is None:
            shuffle = split == 'train'
        meta = CUBMeta()

        self.data_dir = ospj(data_dir, 'images')
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


class CUB(CUBFiles):
    def __init__(self, data_dir, split, shuffle=None):
        super(CUB, self).__init__(data_dir, split, shuffle)

    def get_data(self):
        for fname, label, bbox in super(CUB, self).get_data():
            img = cv2.imread(fname, cv2.IMREAD_COLOR)
            yield [img, label, bbox]
