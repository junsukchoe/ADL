import argparse
import os

from os.path import join as ospj

_DATASET_NAMES = ('CUB', 'ILSVRC')
_METHOD_NAMES = ('CAM', 'ADL')
_ARCH_NAMES = ('resnet50_se', 'vgg_gap')


def set_gpus(args):
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        raise argparse.ArgumentTypeError('GPU id(s) expected.')


def get_data_dir(args):
    args.data_dir = ospj(args.data_dir, args.dataset_name)
    return args


def parse_gating_position(args, number_of_positions=100):
    def to_bool(gating_position_int_list, _number_of_positions):
        gating_position_bool_list = [False] * _number_of_positions
        for i in gating_position_int_list:
            gating_position_bool_list[i] = True
        return gating_position_bool_list

    if args.gating_position:
        args.gating_position = to_bool(args.gating_position,
                                       number_of_positions)
    else:
        args.gating_position = [False] * number_of_positions

    return args


def get_training_configs_per_dataset(args):
    if args.dataset_name == 'CUB':
        args.stepscale = 5.0
        args.number_of_class = 1000 if args.arch_name == 'resnet50_se' else 200
        args.number_of_val = 5794
    elif args.dataset_name == 'ILSVRC':
        args.stepscale = 0.2
        args.number_of_class = 1000
        args.number_of_val = 50000
    else:
        raise KeyError("Unavailable dataset: {}".format(args.dataset_name))
    return args


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu',
                        help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data_dir', help='ILSVRC dataset dir')
    parser.add_argument('--log_dir', help='log directory name')

    parser.add_argument('--epoch', help='max epoch', type=int, default=105)
    parser.add_argument('--final_size', type=int, default=224)
    parser.add_argument('--is_data_format_nhwc', action='store_true')

    parser.add_argument('--dataset_name', type=str, choices=_DATASET_NAMES)
    parser.add_argument('--method_name', type=str, choices=_METHOD_NAMES)
    parser.add_argument('--arch_name', type=str, choices=_ARCH_NAMES)
    parser.add_argument('--gating_position', nargs='+', type=int)
    parser.add_argument('--number_of_class', type=int, default=1000)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--stepscale', type=float, default=1.)

    parser.add_argument('--depth', type=int, default=50)
    parser.add_argument('--mode', help='resnet type', default='se')

    parser.add_argument('--adl_threshold', type=float, default=0.5)
    parser.add_argument('--adl_keep_prob', type=float, default=0.25)
    parser.add_argument('--max_drop_prob', type=float, default=0.1)
    parser.add_argument('--spatial_drop_prob', type=float, default=0.5)

    parser.add_argument('--use_pretrained_model', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--cam_threshold', type=float, default=0.2)
    parser.add_argument('--number_of_val', type=int)
    parser.add_argument('--number_of_cam_curve_interval', type=int, default=7)

    args = parser.parse_args()
    args = get_data_dir(args)
    args = parse_gating_position(args)
    args = get_training_configs_per_dataset(args)

    set_gpus(args)

    return args
