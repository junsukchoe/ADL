import os
import argparse

def get_args():

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', metavar='DIR',
                        default='/workspace/PascalVOC/VOCdevkit/VOC2012/',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg16',
                        help='model architecture: default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=30, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')

    # CASE NAME
    parser.add_argument('--name', type=str, default='test_case')
    parser.add_argument('--task', type=str, default='cls')

    # path
    parser.add_argument('--dataset', type=str, default='CUB')
    parser.add_argument('--data-list', type=str, default='./datalist/CUB/')
    parser.add_argument('--train-list', type=str, default='./datalist/CUB/train.txt')
    parser.add_argument('--test-list', type=str, default='./datalist/CUB/test.txt')
    parser.add_argument('--save-dir', type=str, default='checkpoints/')
    parser.add_argument('--image-save', action='store_true')

    # basic hyperparameters
    parser.add_argument('--LR-decay', type=int, default=30, help='Reducing lr frequency')
    parser.add_argument('--lr-ratio', type=float, default=10)
    parser.add_argument('--nest', action='store_true')

    # ADL
    parser.add_argument('--use-ADL', action='store_true', help='flag for ADL')
    parser.add_argument('--ADL-rate', type=float, default=0.75, help='ADL dropout rate')
    parser.add_argument('--ADL-thr', type=float, default=0.9,
                        help='ADL gamma, threshold ratio to maximum value of attention map')
    parser.add_argument('--ADL-position', nargs='*', help='ADL applied bottleneck')

    parser.add_argument('--cam-thr', type=float, default=0.2, help='cam threshold value')
    parser.add_argument('--cam-curve', action='store_true')

    # bbox
    # data transform
    parser.add_argument('--tencrop', action='store_true')
    parser.add_argument('--resize-size', type=int, default=256, help='validation resize size')
    parser.add_argument('--crop-size', type=int, default=224, help='validation crop size')

    args = parser.parse_args()

    return args