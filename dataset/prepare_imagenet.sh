#!/bin/bash

mkdir -p dataset/ILSVRC/val
tar xvf dataset/ILSVRC2012_img_val.tar -C dataset/ILSVRC/val
mkdir -p dataset/ILSVRC/train
tar xvf dataset/ILSVRC2012_img_train.tar -C dataset/ILSVRC/train
cd dataset/ILSVRC/train
find -type f -name '*.tar' | parallel -P 10 'echo {} && mkdir -p {/.} && tar xf {} -C {/.}'