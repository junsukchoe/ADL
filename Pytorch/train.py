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
from utils.util_eval import *
from utils.util import *

best_epoch = 0
best_acc1 = 0
best_loc1 = 0
loc1_at_best_acc1 = 0
acc1_at_best_loc1 = 0
gtknown_at_best_acc1 = 0
gtknown_at_best_loc1 = 0
writer = None

def main():
    args = get_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1, best_loc1, best_epoch, \
        loc1_at_best_acc1, acc1_at_best_loc1, \
        gtknown_at_best_acc1, gtknown_at_best_loc1
    global writer

    args.gpu = gpu
    log_folder = os.path.join('train_log', args.name)

    if args.gpu == 0:
        writer = SummaryWriter(logdir=log_folder)

    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)

    Logger(os.path.join(log_folder, 'log.log'))
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.dataset == 'CUB':
        num_classes = 200
    elif args.dataset == 'ILSVRC':
        num_classes = 1000
    else:
        raise Exception("Not preferred dataset.")

    if args.arch == 'vgg16':
        model = vgg.vgg16(pretrained=True,
                          num_classes=num_classes)
    elif args.arch == 'vgg16_GAP':
        model = vgg.vgg16_GAP(pretrained=True,
                              num_classes=num_classes)
    elif args.arch == 'vgg16_ADL':
        model = vgg.vgg16_ADL(pretrained=True,
                              num_classes=num_classes,
                              ADL_position=args.ADL_position,
                              drop_rate=args.ADL_rate,
                              drop_thr=args.ADL_thr)
    elif args.arch == 'resnet50_ADL':
        model = resnet.resnet50(pretrained=True,
                                num_classes=num_classes,
                                ADL_position=args.ADL_position,
                                drop_rate=args.ADL_rate,
                                drop_thr=args.ADL_thr)
    elif args.arch == 'resnet50':
        model = resnet.resnet50(pretrained=True,
                                num_classes=num_classes)
    else:
        model = None

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    param_features = []
    param_classifiers = []
    if args.arch.startswith('vgg'):
        for name, parameter in model.named_parameters():
            if 'features.' in name:
                param_features.append(parameter)
            else:
                param_classifiers.append(parameter)

    elif args.arch.startswith('resnet'):
        for name, parameter in model.named_parameters():
            if 'layer4.' in name or 'fc.' in name:
                param_classifiers.append(parameter)
            else:
                param_features.append(parameter)
    else:
        raise Exception("Fail to recognize the architecture")

    optimizer = torch.optim.SGD([
        {'params': param_features, 'lr': args.lr},
        {'params': param_classifiers, 'lr': args.lr * args.lr_ratio}],
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nest)

    # optionally resume from a checkpoint
    if args.resume:
        model, optimizer = load_model(model, optimizer, args)

    cudnn.benchmark = True

    # CUB-200-2011
    train_loader, val_loader, train_sampler = data_loader(args)

    if args.cam_curve:
        cam_curve(val_loader, model, criterion, writer, args)
        return

    if args.evaluate:
        evaluate(val_loader, model, criterion, args)
        return

    if args.gpu == 0:
        print("Batch Size per Tower: %d"%(args.batch_size))
        print(model)

    for epoch in range(args.start_epoch, args.epochs):
        if args.gpu == 0:
            print("===========================================================")
            print("Start Epoch %d ..." % (epoch+1))

        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        val_acc1 = 0
        val_loss = 0
        val_gtloc = 0
        val_loc = 0
        # train for one epoch
        train_acc, train_loss, progress_train = \
                train(train_loader, model, criterion, optimizer, epoch, args)

        if args.gpu == 0:
            progress_train.display(epoch+1)

        # evaluate on validation set
        if args.task == 'cls':
            val_acc1, val_loss = validate(val_loader, model, criterion, epoch, args)

        # evaluate localization on validation set
        elif args.task == 'wsol':
            val_acc1, val_acc5, val_loss, \
            val_gtloc, val_loc = evaluate_loc(val_loader, model, criterion, epoch, args)

        # tensorboard
        if args.gpu == 0:
            writer.add_scalar(args.name + '/train_acc', train_acc, epoch)
            writer.add_scalar(args.name + '/train_loss', train_loss, epoch)
            writer.add_scalar(args.name + '/val_cls_acc', val_acc1, epoch)
            writer.add_scalar(args.name + '/val_loss', val_loss, epoch)
            writer.add_scalar(args.name + '/val_gt_loc', val_gtloc, epoch)
            writer.add_scalar(args.name + '/val_loc1', val_loc, epoch)

        # remember best acc@1 and save checkpoint
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)
        if is_best:
            best_epoch = epoch + 1
            loc1_at_best_acc1 = val_loc
            gtknown_at_best_acc1 = val_gtloc

        if args.task == 'wsol':
            # in case best loc,, Not using this.
            is_best_loc = val_loc > best_loc1
            best_loc1 = max(val_loc, best_loc1)
            if is_best_loc:
                best_epoch = epoch + 1
                acc1_at_best_loc1 = val_acc1
                gtknown_at_best_loc1 = val_gtloc

        if args.gpu == 0:
            print("\nCurrent Best Epoch: %d" %(best_epoch))
            print("Top-1 GT-Known Localization Acc: %.3f \
                   \nTop-1 Localization Acc: %.3f\
                   \nTop-1 Classification Acc: %.3f" % \
                  (gtknown_at_best_acc1, loc1_at_best_acc1, best_acc1))
            print("\nEpoch %d finished." % (epoch+1))

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            saving_dir = os.path.join(log_folder)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, saving_dir)

    if args.gpu == 0:
        save_train(best_acc1, loc1_at_best_acc1, gtknown_at_best_acc1,
                   best_loc1, acc1_at_best_loc1, gtknown_at_best_loc1, args)

        print("===========================================================")
        print("Start Evaluation on Best Checkpoint ...")

    args.resume = os.path.join(log_folder, 'model_best.pth.tar')
    model, _ = load_model(model, optimizer, args)
    evaluate(val_loader, model, criterion, args)
    cam_curve(val_loader, model, criterion, writer, args)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = SumEpochMeter('Time', ':6.3f')
    data_time = SumEpochMeter('Data', ':6.3f')
    losses = AverageEpochMeter('Loss', ':.4e')
    top1 = AverageEpochMeter('Top-1 Classification Acc')
    top5 = AverageEpochMeter('Top-5 Classification Acc')
    learning_rate = AverageEpochMeter('Learning Rate:', ':.1e')
    progress = ProgressEpochMeter(
        len(train_loader),
        [batch_time, data_time, losses, learning_rate, top1, top5],
        prefix="\nTraining Phase: ")

    for param_group in optimizer.param_groups:
        learning_rate.update(param_group['lr'])
        break

    # switch to train mode
    model.train()
    end = time.time()

    means = [.485, .456, .406]
    stds = [.229, .224, .225]
    means = torch.reshape(torch.tensor(means), (1, 3, 1, 1)).cuda(args.gpu)
    stds = torch.reshape(torch.tensor(stds), (1, 3, 1, 1)).cuda(args.gpu)

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        adl_maps = list()

        with torch.no_grad():
            if 'ADL' in args.arch:
                if i < 5:
                    image_ = images.clone().detach() * stds + means
                    for name, module in model.module.named_modules():
                        if 'resnet' in args.arch:
                            if isinstance(module, resnet.ADL):
                                adl_maps.append(module.get_maps())
                        elif 'vgg' in args.arch:
                            if isinstance(module, vgg.ADL):
                                adl_maps.append(module.get_maps())
                    b, _, h, w = image_.shape
                    image_ = F.interpolate(image_, (h//2, w//2))
                    image_set = vutils.make_grid(image_[:16], nrow=16)
                    for idx, (attention, drop_mask) in enumerate(adl_maps):
                        attention = F.interpolate(attention, (h//2, w//2),
                                                  mode='bilinear', align_corners=False)
                        drop_mask = F.interpolate(drop_mask, (h//2, w//2),
                                                  mode='bilinear', align_corners=False)
                        attention = vutils.make_grid(attention[:16], nrow=16,
                                                     normalize=True, scale_each=True)
                        drop_mask = vutils.make_grid(drop_mask[:16], nrow=16,
                                                     normalize=True, scale_each=True)

                        if idx == 0:
                            compare_set = torch.cat((image_set, attention, drop_mask), dim=1)
                        else:
                            compare_set = torch.cat((compare_set, attention, drop_mask), dim=1)
                    if args.gpu == 0:
                        writer.add_image(args.name+'/ADL_'+str(args.gpu), compare_set, epoch * 5 + i)


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return top1.avg, losses.avg, progress


if __name__ == '__main__':
    main()
