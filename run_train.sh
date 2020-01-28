#!/bin/bash

# make sure the name of dataset directory is same with `dataset_name`
# ex: /notebooks/dataset/ILSVRC | /notebooks/dataset/CUB

# gpu ids are sepearted with comma. ex: --gpu 0,1,2,3,4,5,6,7 | --gpu 2,5
# dataset_name: (ILSVRC, CUB)
# method_name: (CAM, ADL)
# arch_name: (resnet50_se, vgg_gap)

python train.py \
      --gpu 1 \
      --data_dir /notebooks/dataset/ \
      --dataset_name CUB \
      --method_name ADL \
      --arch_name resnet50_se \
      --base_lr 0.1 \
      --log_dir ResNet50SE_CUB \
      --use_pretrained_model \
      --batch 32 \
      --depth 50 \
      --gating_position 31 41 5 \
      --adl_keep_prob 0.25 \
      --adl_threshold 0.90 
