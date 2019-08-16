#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: TinyImagenet.py

# This code is mainly borrowed from the official example codes of tensorpack library.
# https://github.com/ppwwyyxx/tensorpack/tree/master/tensorpack/dataflow/dataset

# Revised by Junsuk Choe <skykite@yonsei.ac.kr>
# Dataflow implementation for Tiny ImageNet
# https://tiny-imagenet.herokuapp.com/

import os
import tarfile
import numpy as np
import tqdm
import random
import cv2

#from ...utils import logger
#from ..base import RNGDataFlow
from tensorpack.utils import logger
from tensorpack.dataflow.base import RNGDataFlow

__all__ = ['TinyImagenetMeta', 'TinyImagenet', 'TinyImagenetFiles']

class TinyImagenetMeta(object):
    """
    Provide methods to access metadata for tinyImagenet dataset.
    """

    def __init__(self, dir=None):
        self.dir = dir
        f = os.path.join(self.dir, 'wnids.txt')

    def get_synset_words_1000(self, classname):
        """
        Returns:
            dict: {cls_number: cls_name}
        """
        fname = os.path.join(self.dir,'labels', classname, 'words.txt')
        
        assert os.path.isfile(fname)
        lines = [x.strip().split('\t') for x in open(fname).readlines()]
        return dict(lines)

    def get_synset_1000(self, classname):
        """
        Returns:
            dict: {cls_number: synset_id}
        """
        fname = os.path.join(self.dir,'labels',classname, 'wnids.txt')
        assert os.path.isfile(fname)
        lines = [x.strip() for x in open(fname).readlines()]
        return dict(enumerate(lines))

    def get_image_list(self, name, classname, dir_structure='original'):
        """
        Args:
            name (str): 'train' or 'val' or 'test'
            dir_structure (str): same as in :meth:`TinyImagenet.__init__()`.
        Returns:
            list: list of (image filename, label)
        """
        assert name in ['train', 'val', 'test']
        assert dir_structure in ['original', 'train']

        fname = os.path.join(self.dir, 'labels', classname, name + '.txt')
        assert os.path.isfile(fname), fname
        with open(fname) as f:
            ret = []
            for line in f.readlines():
                temp = line.strip().split()
                if len(temp) == 2:
                    name, cls = line.strip().split() # train
                    xa = 1 # dummy
                    ya = 1 # dummy
                    xb = 1 # dummy
                    yb = 1 # dummy
                else:
                    name, cls, xa, ya, xb, yb = line.strip().split() # val
                    xa = float(xa)
                    ya = float(ya)
                    xb = float(xb)
                    yb = float(yb)
                bbox = np.array([(xa, ya), (xb, yb)], dtype=np.float)  
                cls = int(cls)
                ret.append((name.strip(), cls, bbox))
        assert len(ret), fname
        return ret


def _guess_dir_structure(dir):
    subdir = os.listdir(dir)[0]
    # find a subdir starting with 'n'
    if subdir.startswith('n') and \
            os.path.isdir(os.path.join(dir, subdir)):
        dir_structure = 'train'
    else:
        dir_structure = 'original'
    logger.info(
        "Assuming directory {} has {} structure.".format(
            dir, dir_structure))
    return dir_structure

class TinyImagenetFiles(RNGDataFlow):
    """
    Same as :class:`TinyImagenet`, but produces filenames of the images instead of nparrays.
    """
    def __init__(self, dir, name, classname, meta_dir=None,
                 shuffle=None, dir_structure=None):
        """
        Same as in :class:`TinyImagenet`.
        """
        assert name in ['train', 'test', 'val'], name
        assert os.path.isdir(dir), dir
        self.full_dir = os.path.join(dir, name)
        #self.full_dir = dir
        self.name = name
        assert os.path.isdir(self.full_dir), self.full_dir
        assert meta_dir is None or os.path.isdir(meta_dir), meta_dir
        if shuffle is None:
            shuffle = name == 'train'
        self.shuffle = shuffle

        if name == 'train':
            dir_structure = 'train'
        if dir_structure is None:
            dir_structure = _guess_dir_structure(self.full_dir)

        meta_dir = dir 
        meta = TinyImagenetMeta(meta_dir)
        self.imglist = meta.get_image_list(name, classname, dir_structure)

        for fname, _, _  in self.imglist[:10]:
            fname = os.path.join(self.full_dir, fname)
            assert os.path.isfile(fname), fname
    
    def size(self):
        return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            fname, label, bbox = self.imglist[k]
            fname = os.path.join(self.full_dir, fname)
            yield [fname, label, bbox]


class TinyImagenet(TinyImagenetFiles):
    """
    Produces uint8 TinyImagenet images of shape [h, w, 3(BGR)], and a label between [0, 999].
    """
    def __init__(self, dir, name, classname, meta_dir=None,
                 shuffle=None, dir_structure=None):
        """
        Args:
            dir (str): A directory containing a subdir named ``name``, where the
                original ``TinyImagenet_img_{name}.tar`` gets decompressed.
            name (str): 'train' or 'val' or 'test'.
            shuffle (bool): shuffle the dataset.
                Defaults to True if name=='train'.
            dir_structure (str): The directory structure of 'val' and 'test' directory.
                'original' means the original decompressed
                directory, which only has list of image files (as below).
                If set to 'train', it expects the same two-level
                directory structure simlar to 'train/'.
                By default, it tries to automatically detect the structure.

        Examples:

        When `dir_structure=='original'`, `dir` should have the following structure:

        .. code-block:: none

            dir/
              train/
                n02134418/
                  images/
                    n02134418_18.JPEG
                    ...
                  ...
                ...
              val/
                val_0001.JPEG
                ...
              test/
                test_0001.JPEG
                ...
                
        """
        super(TinyImagenet, self).__init__(dir, name, classname, meta_dir, shuffle, dir_structure)
        
        
    def get_data(self):
        for fname, label, bbox in super(TinyImagenet, self).get_data():
            img = cv2.imread(fname, cv2.IMREAD_COLOR)
            assert img is not None, fname
            
            yield [img, label, bbox]
            



if __name__ == '__main__':
    meta = TinyImagenetMeta()

    ds = TinyImagenet('/home/wyx/data/fake_TinyImagenet/', 'train', shuffle=False)
    ds.reset_state()

    for k in ds.get_data():
        from IPython import embed
        embed()
        break
