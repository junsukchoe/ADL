# -*- coding: utf-8 -*-
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils

from network import vgg, resnet
from tensorboardX import SummaryWriter
from utils.util_args import get_args
from utils.util_acc import accuracy, adjust_learning_rate, \
    save_checkpoint, AverageEpochMeter, SumEpochMeter, \
    ProgressEpochMeter, calculate_IOU, Logger
from utils.util_loader import data_loader
from utils.util_bbox import *
from utils.util_cam import *
from utils.util import *

def validate(val_loader, model, criterion, epoch, args):
    global writer
    batch_time = SumEpochMeter('Time', ':6.3f')
    losses = AverageEpochMeter('Loss', ':.4e')
    top1 = AverageEpochMeter('Top-1 Classification Acc')
    top5 = AverageEpochMeter('Top-5 Classification Acc')
    progress = ProgressEpochMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix="\nValidation Phase: ")

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target, image_ids) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if args.gpu == 0:
            progress.display(epoch+1)

    return top1.avg, losses.avg


def evaluate_loc(val_loader, model, criterion, epoch, args):
    batch_time = SumEpochMeter('Time')
    losses = AverageEpochMeter('Loss')
    top1 = AverageEpochMeter('Top-1 Classification Acc')
    top5 = AverageEpochMeter('Top-5 Classification Acc')
    GT_loc = AverageEpochMeter('Top-1 GT-Known Localization Acc')
    top1_loc = AverageEpochMeter('Top-1 Localization Acc')
    progress = ProgressEpochMeter(
        len(val_loader),
        [batch_time, losses, GT_loc, top1_loc, top1, top5],
        prefix="\nValidation Phase: ")

    # image 개별 저장할 때 필요
    image_names = get_image_name(args.test_list)
    gt_bbox = load_bbox_size(dataset_path=args.data_list,
                             resize_size = args.resize_size,
                             crop_size = args.crop_size)

    cnt = 0
    cnt_false = 0
    hit_known = 0
    hit_top1 = 0

    means = [0.485, .456, .406]
    stds = [.229, .224, .225]
    means = torch.reshape(torch.tensor(means), (1, 3, 1, 1))
    stds = torch.reshape(torch.tensor(stds), (1, 3, 1, 1))


    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target, image_ids) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True)
            image_ids = image_ids.data.cpu().numpy()
            output = model(images)
            loss = criterion(output, target)

            # Get acc1, acc5 and update
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            wrongs = [c == 0 for c in correct.cpu().numpy()][0]

            # original image in tensor format
            image_ = images.clone().detach().cpu() * stds + means

            # cam image in tensor format
            cam = get_cam(model=model, target=target, args=args)

            # generate tensor base blend image
            # blend_tensor = generate_blend_tensor(image_, cam)

            # generate bbox
            blend_tensor = torch.zeros_like(image_)
            image_ = image_.clone().detach().cpu().numpy().transpose(0, 2, 3, 1)
            cam_ = cam.clone().detach().cpu().numpy().transpose(0, 2, 3, 1)

            # reverse the color representation(RGB -> BGR) and Opencv format
            image_ = image_[:, :, :, ::-1] * 255
            # cam_ = cam_[:, :, :, ::-1]
            for j in range(images.size(0)):

                estimated_bbox, blend_bbox = generate_bbox(image_[j],
                                                           cam_[j],
                                                           gt_bbox[image_ids[j]],
                                                           args.cam_thr)

                # reverse the color representation(RGB -> BGR) and reshape
                if args.gpu == 0:
                    blend_bbox = blend_bbox[:, :, ::-1] / 255.
                    blend_bbox = blend_bbox.transpose(2, 0, 1)
                    blend_tensor[j] = torch.tensor(blend_bbox)

                # calculate IOU for WSOL
                IOU = calculate_IOU(gt_bbox[image_ids[j]], estimated_bbox)
                if IOU >= 0.5:
                    hit_known += 1
                    if not wrongs[j]:
                        hit_top1 += 1
                if wrongs[j]:
                    cnt_false += 1

                cnt += 1

            # save the tensor
            if args.gpu == 0 and i < 1 and not args.cam_curve:
                save_images('results', epoch, i, blend_tensor, args)

            if args.gpu == 0 and args.evaluate and not args.cam_curve:
                save_images('results_best', epoch, i, blend_tensor, args)


            loc_gt = hit_known / cnt * 100
            loc_top1 = hit_top1 / cnt * 100

            GT_loc.update(loc_gt, images.size(0))
            top1_loc.update(loc_top1, images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

        if args.gpu == 0:
            progress.display(epoch+1)

    torch.cuda.empty_cache()

    return top1.avg, top5.avg, losses.avg, GT_loc.avg, top1_loc.avg

def save_images(folder_name, epoch, i, blend_tensor, args):
    saving_folder = os.path.join('train_log', args.name, folder_name)
    if not os.path.isdir(saving_folder):
        os.makedirs(saving_folder)
    file_name = 'HEAT_TEST_{}_{}.jpg'.format(epoch+1, i)
    saving_path = os.path.join(saving_folder, file_name)
    if args.gpu == 0:
        vutils.save_image(blend_tensor, saving_path)

def save_train(best_acc1, loc1_at_best_acc1, gtknown_at_best_acc1,
               best_loc1, acc1_at_best_loc1, gtknown_at_best_loc1, args):
    with open(os.path.join('train_log', args.name, args.name + '.txt'), 'w') as f:
        line = 'Best Acc1: %.3f, Loc1: %.3f, GT: %.3f\n' % \
               (best_acc1, loc1_at_best_acc1, gtknown_at_best_acc1)
        f.write(line)
        line = 'Best Loc1: %.3f, Acc1: %.3f, GT: %.3f' % \
               (best_loc1, acc1_at_best_loc1, gtknown_at_best_loc1)
        f.write(line)


def cam_curve(val_loader, model, criterion, writer, args):
    cam_thr_list = [round(i * 0.01, 2) for i in range(0, 100, 5)]
    thr_loc = {}

    args.cam_curve = True

    for step, i in enumerate(cam_thr_list):
        args.cam_thr = i
        if args.gpu == 0:
            print('\nCAM threshold: %.2f' % args.cam_thr)
        val_acc1, val_acc5, val_loss, \
        val_gtloc, val_loc = evaluate_loc(val_loader, model, criterion, 1, args)

        thr_loc[i] = [val_acc1, val_acc5, val_loc, val_gtloc]
        if args.gpu == 0:
            writer.add_scalar(args.name + '/cam_curve', val_loc, step)
            writer.add_scalar(args.name + '/cam_curve', val_gtloc, step)

    with open(os.path.join('train_log', args.name, 'cam_curve_results.txt'), 'w') as f:
        for i in cam_thr_list:
            line = 'CAM_thr: %.2f Acc1: %3f Acc5: %.3f Loc1: %.3f GTloc: %.3f \n' % \
                   (i, thr_loc[i][0], thr_loc[i][1], thr_loc[i][2], thr_loc[i][3])
            f.write(line)

    return


def evaluate(val_loader, model, criterion, args):
    args.evaluate = True

    val_acc1, val_acc5, val_loss, \
    val_gtloc, val_loc = evaluate_loc(val_loader, model, criterion, 0, args)
