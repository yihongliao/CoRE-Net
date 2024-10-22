import sys
import os
import socket
import getpass
import argparse

import glog as log

import numpy as np
import torch
from utils.loss import dice_bce_loss
from networks.corenet import CoRE_Net
from networks.framework import MyFrame
from utils.data_selftrain import ImageFolder as ImageFolder_selftrain
import utils.infer as my_infer
import cv2

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--exp-dir', default='./test/', type=str,
                        help='directory to save results and logs')
    parser.add_argument('--data-root', default='./data/LVD2021', type=str,
                        help='directory to load training or testing data')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='batch size')

    parser.add_argument('--total-epoch', default=30, type=int,
                        help='if early stopping does not perform, it will run the max epoch')
    parser.add_argument('--total-inner-epoch', default=10, type=int,
                        help='the number of inner iteration epoch during self-training')
    parser.add_argument('--dataset', default='36_Holly_labels', type=str,
                        help='dataset to apply')

    parser.add_argument('--if-threshold', default='0.99,0.05', type=str,
                        help='the inference threshold of the head1 output, '
                             '[0] higher border, [1] lower border')
    parser.add_argument('--load-pretrained-path', type=str,
                        help='the loading path of pretrained model parameters')
    parser.add_argument('--mask-point-on', action='store_true',
                        help='whether to use point refiner.')
    parser.add_argument('--confidence-on', action='store_true',
                        help='whether to use CMM.')
    parser.add_argument('--point-correction-on', action='store_true',
                        help='whether to use PCM.')
    parser.add_argument('--num-subdivision-points', default=28*28, type=int,
                        help='number of most uncertain points selected')
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup-steps", default=10, type=int,
                        help="Step of training to perform learning rate warmup for.")

    parser.add_argument('--initial-epoch-loss', default=100000, type=float,
                        help='initial epoch loss, a large number, so as to minimize it')
    parser.add_argument('--num-early-stop', default=20, type=int,
                        help='stop training when the loss is not optimized for this certain epochs')
    parser.add_argument('--num-update-lr', default=10, type=int,
                        help='break when the loss is not optimized for this certain epochs and lr is lower than 5e-7')
    parser.add_argument('--lambda-dice-iou-loss', default=0.5, type=float,
                        help='the penalty of dice-iou loss')
    
    parser.add_argument('--tensorboard', action='store_true',
                        help='whether to record in tensorboardX or not '
                             'If set to True, we will record all output samples during training')
    parser.add_argument('--seed', default=1234, type=int,
                        help='random seed')
    parser.add_argument('--ssh', action='store_true',
                        help='whether or not we are executing command via ssh. '
                             'If set to True, we will not print anything to screen and only redirect them to log file')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))

def main(args):
    log.info('Host: {}, user: {}, CUDA_VISIBLE_DEVICES: {}, cwd: {}'.format(
        socket.gethostname(), getpass.getuser(), os.environ.get('CUDA_VISIBLE_DEVICES', ''), os.getcwd()))

    log.info('torch version: {}'.format(torch.__version__))
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info('Called with args:')
    print_args(args)

    # build model
    log.info('building model with evalmode')

    args.mask_point_on = args.confidence_on or args.point_correction_on
    solver = MyFrame(CoRE_Net, dice_bce_loss, args, evalmode=True, pointmode=args.mask_point_on)
    solver_save_path = args.load_pretrained_path

    log.info('loading retrained model')
    solver.load(solver_save_path)

    log.info('testing data')
    testset = ImageFolder_selftrain(root_path=args.data_root, datasets=args.dataset, mode='test', is_random=False)
    log.info('done!')

    log.info('========start inference=======')
    trainset = my_infer.infer_labels(args, testset, testset, solver, 0)
    log.info('========inference end=======')

    for i, img in enumerate(trainset.masks):
        cv2.imwrite(xargs.exp_dir + str(i) + '.png', img)


if __name__ == '__main__':
    xargs = parse_args()

    os.makedirs(xargs.exp_dir, exist_ok=True)

    main(xargs)