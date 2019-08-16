#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: CUB200.py

# This code is mainly borrowed from the official example codes of tensorpack library.
# https://github.com/ppwwyyxx/tensorpack/tree/master/tensorpack/dataflow/dataset

# Revised by Junsuk Choe <skykite@yonsei.ac.kr>
# Dataflow implementation for Tiny CUB200
# https://tiny-CUB200.herokuapp.com/

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

__all__ = ['CUB200Meta', 'CUB200', 'CUB200Files']

class CUB200Meta(object):
    """
    Provide methods to access metadata for CUB200 dataset.
    """

    def __init__(self, dir=None):
        self.dir = dir
        f = os.path.join(self.dir, 'wnids.txt')

    def get_synset_words_1000(self, classname):
        """
        Returns:
            dict: {cls_number: cls_name}
        """
        fname = os.path.join(self.dir, 'words.txt')
        
        assert os.path.isfile(fname)
        lines = [x.strip().split(' ', 1) for x in open(fname).readlines()]
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
            dir_structure (str): same as in :meth:`CUB200.__init__()`.
        Returns:
            list: list of (image filename, label)
        """
        assert name in ['train', 'val', 'test']

        fname = os.path.join(self.dir, 'labels', classname, name + '.txt')
        assert os.path.isfile(fname), fname
        with open(fname) as f:
            ret = []
            for line in f.readlines():
                temp = line.strip().split()
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

'''
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
'''
class CUB200Files(RNGDataFlow):
    """
    Same as :class:`CUB200`, but produces filenames of the images instead of nparrays.
    """
    def __init__(self, dir, name, classname, meta_dir=None,
                 shuffle=None, dir_structure=None):
        """
        Same as in :class:`CUB200`.
        """
        assert name in ['train', 'test', 'val'], name
        assert os.path.isdir(dir), dir
        self.full_dir = dir
        self.name = name
        assert os.path.isdir(self.full_dir), self.full_dir
        assert meta_dir is None or os.path.isdir(meta_dir), meta_dir
        if shuffle is None:
            shuffle = name == 'train'
        self.shuffle = shuffle

        if name == 'train':
            dir_structure = 'train'
        #if dir_structure is None:
        #    dir_structure = _guess_dir_structure(self.full_dir)

        meta_dir = dir 
        meta = CUB200Meta(meta_dir)
        self.imglist = meta.get_image_list(name, classname, dir_structure)

        for fname, _, _ in self.imglist[:10]:
            fname = os.path.join(self.full_dir, 'images', fname)
            assert os.path.isfile(fname), fname
    
    def size(self):
        return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            fname, label, bbox = self.imglist[k]
            fname = os.path.join(self.full_dir, 'images', fname)
            yield [fname, label, bbox]


class CUB200(CUB200Files):
    """
    Produces uint8 CUB200 images of shape [h, w, 3(BGR)], and a label between [0, 999].
    """
    def __init__(self, dir, name, classname, meta_dir=None,
                 shuffle=None, dir_structure=None):
        """
        Args:
            dir (str): A directory containing a subdir named ``name``, where the
                original ``CUB200_img_{name}.tar`` gets decompressed.
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
                    n02134418_18.JPEG
                    ...
                ...
              val/
                val_0001.JPEG
                ...
              test/
                test_0001.JPEG
                ...
                
        """
        super(CUB200, self).__init__(dir, name, classname, meta_dir, shuffle, dir_structure)
        
        
    def get_data(self):
        for fname, label, bbox in super(CUB200, self).get_data():
            img = cv2.imread(fname, cv2.IMREAD_COLOR)
            assert img is not None, fname
            
            yield [img, label, bbox]
            



if __name__ == '__main__':
    meta = CUB200Meta()

    ds = CUB200('/home/wyx/data/fake_CUB200/', 'train', shuffle=False)
    ds.reset_state()

    for k in ds.get_data():
        from IPython import embed
        embed()
        break
