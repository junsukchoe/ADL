#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: utils_loc.py

# Written by Junsuk Choe <skykite@yonsei.ac.kr>
# Function Code for visualizing heatmap of learned CNNs.
# Including CAM and Grad-CAM.

import cv2
import sys
import numpy as np
import os
import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.utils import viz
from tensorpack.utils.fs import mkdir_p

from utils import get_data
from dataflow import Imagenet, CUB200

def cam(model, option, gradcam=False, flag=None):
    model_file = option.load
    data_dir = option.data
    if option.imagenet:
        valnum = 50000
    elif option.cub:
        valnum = 5794

    ds = get_data('val', option)

    pred_config = PredictConfig(
        model=model,
        session_init=get_model_loader(model_file),
        input_names=['input', 'label','bbox'],
        output_names=
            ['wrong-top1', 'top5', 'actmap', 'grad'],
        return_input=True
    )

    if option.imagenet:
        meta = Imagenet.ImagenetMeta(dir=option.data). \
                get_synset_words_1000(option.dataname)
        meta_labels = Imagenet.ImagenetMeta(dir=option.data). \
                get_synset_1000(option.dataname)
    elif option.cub:
        meta = CUB200.CUB200Meta(dir=option.data). \
                get_synset_words_1000(option.dataname)
        meta_labels = CUB200.CUB200Meta(dir=option.data). \
                get_synset_1000(option.dataname)
    pred = SimpleDatasetPredictor(pred_config, ds)

    cnt = 0
    cnt_false = 0
    hit_known = 0
    hit_top1 = 0

    index = int(option.locthr*100)

    if option.camrelu:

        dirname = os.path.join(
            'train_log',option.logdir,'result_camrelu',str(index))
    else:

        dirname = os.path.join(
            'train_log',option.logdir,'result_norelu',str(index))

    if not os.path.isdir(dirname):
        mkdir_p(dirname)

    for inp, outp in pred.get_result():
        images, labels, bbox = inp

        if gradcam:
            wrongs, top5, convmaps, grads_val = outp
            batch = wrongs.shape[0]
            if option.chlast:
                NUMBER,HEIGHT,WIDTH,CHANNEL = np.shape(convmaps)
            else:
                NUMBER,CHANNEL,HEIGHT,WIDTH = np.shape(convmaps)
            if not option.chlast:
                grads_val = np.transpose(grads_val, [0,2,3,1])
            W = np.mean(grads_val, axis=(1,2))
            if option.chlast:
                convmaps = np.transpose(convmaps, [0,3,1,2])
        else:
            wrongs, top5, convmaps, W = outp
            batch = wrongs.shape[0]
            NUMBER,CHANNEL,HEIGHT,WIDTH = np.shape(convmaps)

        for i in range(batch):
            gxa = int(bbox[i][0][0])
            gya = int(bbox[i][0][1])
            gxb = int(bbox[i][1][0])
            gyb = int(bbox[i][1][1])

            # generating heatmap
            weight = W[i]   # c x 1

            convmap = convmaps[i, :, :, :]  # c x h x w
            mergedmap = np.matmul(weight, convmap.reshape((CHANNEL, -1))). \
                            reshape(HEIGHT, WIDTH)
            if option.camrelu: mergedmap = np.maximum(mergedmap, 0)
            mergedmap = cv2.resize(mergedmap,
                            (option.final_size, option.final_size))
            heatmap = viz.intensity_to_rgb(mergedmap, normalize=True)
            blend = images[i] * 0.5 + heatmap * 0.5

            # initialization for boundary box
            bbox_img = images[i]
            bbox_img = bbox_img.astype('uint8')
            heatmap = heatmap.astype('uint8')
            blend = blend.astype('uint8')

            # thresholding heatmap
            # For computation efficiency, we revise this part by directly using mergedmap.
            # gray_heatmap = cv2.cvtColor(heatmap,cv2.COLOR_RGB2GRAY)
            gray_heatmap = mergedmap
            gray_heatmap /= gray_heatmap.max()
            gray_heatmap = gray_heatmap * 255
            gray_heatmap = gray_heatmap.astype('uint8')

            th_value = np.max(gray_heatmap)*option.locthr

            _, thred_gray_heatmap = \
                        cv2.threshold(gray_heatmap,int(th_value),
                                                255,cv2.THRESH_TOZERO)
            _, contours, _ = \
                        cv2.findContours(thred_gray_heatmap,
                                cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            # calculate bbox coordinates
            rect = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                rect.append([x,y,w,h])

            if rect == []:
                estimated_box = [0,0,1,1] #dummy
            else:
                x,y,w,h = large_rect(rect)
                estimated_box = [x,y,x+w,y+h]
                cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 255, 0), 2)


            cv2.rectangle(bbox_img, (gxa, gya), (gxb, gyb), (0, 0, 255), 2)
            gt_box = [gxa,gya,gxb,gyb]

            IOU_ = bb_IOU(estimated_box, gt_box)

            if IOU_ > 0.5 or IOU_ == 0.5:
                hit_known = hit_known + 1

            if (IOU_ > 0.5 or IOU_ == 0.5) and not wrongs[i]:
                hit_top1 = hit_top1 + 1

            if wrongs[i]:
                cnt_false += 1

            concat = np.concatenate((bbox_img, heatmap, blend), axis=1)
            classname = meta[meta_labels[labels[i]]].split(',')[0]

            if cnt < 500:
                if option.camrelu:
                    cv2.imwrite(
                        'train_log/{}/result_camrelu/{}/cam{}-{}.jpg'. \
                            format(option.logdir, index, cnt, classname),
                            concat)
                else:

                    cv2.imwrite(
                        'train_log/{}/result_norelu/{}/cam{}-{}.jpg'. \
                            format(option.logdir, index, cnt, classname),
                            concat)

            cnt += 1
            if cnt == valnum:
                if option.camrelu:

                    fname = 'train_log/{}/result_camrelu/{}/Loc.txt'. \
                            format(option.logdir, index)
                else:

                    fname = 'train_log/{}/result_norelu/{}/Loc.txt'. \
                            format(option.logdir, index)
                f = open(fname, 'w')
                acc_known = hit_known/cnt
                acc_top1 = hit_top1/cnt
                top1_acc = 1 - cnt_false / (cnt)
                if option.camrelu: print ("\nGRADCAM (use relu)")
                else: print ("\nCAM (do not use relu)")
                print ('Flag: {}\nCAM Threshold: {}\nGT-known Loc: {} \
                        \nTop-1 Loc: {}\nTop-1 Acc: {}' \
                        .format(flag,option.locthr,acc_known,acc_top1,top1_acc))
                line = 'GT-known Loc: {}\nTop-1 Loc: {}\nTop-1 Acc: {}'. \
                        format(acc_known,acc_top1,top1_acc)
                f.write(line)
                f.close()
                return


def bb_IOU(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def large_rect(rect):
    # find largest recteangles
    large_area = 0
    target = 0
    for i in range(len(rect)):
        area = rect[i][2]*rect[i][3]
        if large_area < area:
            large_area = area
            target = i

    x = rect[target][0]
    y = rect[target][1]
    w = rect[target][2]
    h = rect[target][3]

    return x, y, w, h
