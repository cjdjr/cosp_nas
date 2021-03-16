import os
import time
import glob
import numpy as np
import pickle
import torch
import logging
import argparse
import torch
import random
import json
import pprint as pp
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger
import torch.distributed as dist
import collections
import sys
sys.setrecursionlimit(10000)
sys.path.append("..")

this_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir  = "{}/..".format(this_dir)
sys.path.insert(0, lib_dir)

import functools
print = functools.partial(print, flush=True)

from utils import torch_load_cpu, get_inner_model, set_random_seed
from supernet.network import ShuffleNetV2_OneShot
from evaluator.imagenet_dataset import init_evaluator
# from reinforce_baselines import NoBaseline, RolloutBaseline, WarmupBaseline, CriticBaseline, ExponentialBaseline
# from train import train_epoch, validate
# from subnet_dataset import SubnetDataset

# from tester import set_dataprovider

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--flops-limit', type=float, default=330 * 1e6)
    parser.add_argument('--method',type=str,default="COSP")
    parser.add_argument('--supernet',type=str,default="ShuffleNetV2")

    # Evaluator
    parser.add_argument('--max-train-iters', type=int, default=200)
    parser.add_argument('--max-test-iters', type=int, default=40)
    parser.add_argument('--train-batch-size', type=int, default=128)
    parser.add_argument('--test-batch-size', type=int, default=200)
    parser.add_argument('--evaluator_seed', type=int, default=0)

    # Data
    parser.add_argument('--batch_size', type=int, default=16, help='Number of instances per batch during training')
    parser.add_argument('--epoch_size', type=int, default=128, help='Number of instances per epoch during training')
    parser.add_argument('--val_size', type=int, default=100,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--val_dataset', type=str, default=None, help='Dataset file to use for validation')
    parser.add_argument('--val_sample', type=int, default=48,
                        help='Number of instances used for reporting validation performance')
    

    # EA
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--select-num', type=int, default=10)
    parser.add_argument('--population-num', type=int, default=50)
    parser.add_argument('--m_prob', type=float, default=0.1)
    parser.add_argument('--crossover-num', type=int, default=25)
    parser.add_argument('--mutation-num', type=int, default=25)

    # COSP_Model
    # parser.add_argument('--model', default='attention', help="Model, 'attention' (default) or 'pointer'")
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--feed_forward_hidden', type=int, default=512, help='Dimension of feed_forward_hidden in Enc')
    
    parser.add_argument('--n_encode_layers', type=int, default=2,
                        help='Number of layers in the encoder/critic network')
    parser.add_argument('--tanh_clipping', type=float, default=10.,
                        help='Clip the parameters to within +- this value using tanh. '
                             'Set to 0 to not perform any clipping.')
    parser.add_argument('--normalization', default='batch', help="Normalization type, 'batch' (default) or 'instance'")

    # Training
    parser.add_argument('--lr_model', type=float, default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=1.0, help='Learning rate decay per epoch')
    parser.add_argument('--eval_only', action='store_true', help='Set this value to only evaluate model')
    parser.add_argument('--n_epochs', type=int, default=100, help='The number of epochs to train')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
    parser.add_argument('--eval_seed', type=int, type=int,default=0, help='Random seed used in validate')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--exp_beta', type=float, default=0.8,
                        help='Exponential moving average baseline decay (default 0.8)')
    parser.add_argument('--baseline', default="exponential",
                        help="Baseline to use: 'rollout', 'critic' or 'exponential'. Defaults to no baseline.")
    parser.add_argument('--bl_alpha', type=float, default=0.05,
                        help='Significance in the t-test for updating rollout baseline')
    parser.add_argument('--bl_warmup_epochs', type=int, default=None,
                        help='Number of epochs to warmup the baseline, default None means 1 for rollout (exponential '
                             'used for warmup phase), 0 otherwise. Can only be used with rollout baseline.')


    # Misc
    parser.add_argument('--log_step', type=int, default=1, help='Log info every log_step steps')

    parser.add_argument('--run_name', default='run', help='Name to identify the run')
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--load_path', help='Path to load model parameters and optimizer state from')
    parser.add_argument('--resume', help='Resume from previous checkpoint file')
    parser.add_argument('--no_tensorboard', action='store_true', help='Disable logging TensorBoard files')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--no_output', action='store_true')
    parser.add_argument('--local_rank', default=-1, type=int,help='node rank for distributed training')

    args = parser.parse_args()
    return args

def main():

    args = get_args()

    t = time.time()

    args.use_cuda = torch.cuda.is_available() and not args.no_cuda
    args.run_name = "{}_{}".format(args.run_name, time.strftime("%Y%m%dT%H%M%S"))
    args.n_layer = 20
    args.n_op = 4
    args.save_dir = os.path.join(
        args.output_dir,
        args.run_name
    )
    print(args.save_dir)

    # from IPython import embed
    # embed()
    if not args.no_output:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
    if args.bl_warmup_epochs is None:
        args.bl_warmup_epochs = 1 if args.baseline == 'rollout' else 0
    assert (args.bl_warmup_epochs == 0) or (args.baseline == 'rollout')
    assert args.epoch_size % args.batch_size == 0, "Epoch size must be integer multiple of batch size!"

    if args.supernet == "ShuffleNetV2":
        supernet = ShuffleNetV2_OneShot()
        supernet = torch.nn.DataParallel(supernet).cuda()
        supernet_state_dict = torch.load(
            'checkpoint-latest.pth.tar')['state_dict']
        supernet.load_state_dict(supernet_state_dict)

    else:
        print("We don't support this supernet now.")
        return

    if args.method == "COSP":
        from search.COSP import COSPSearcher
        searcher = COSPSearcher(args,supernet)

    elif args.method == "EA":
        from search.EA import EvolutionSearcher
        searcher = EvolutionSearcher(args,supernet)

    else:
        print("We don't support this method now.")
        return

    init_evaluator(args.max_train_iters,args.train_batch_size,args.max_test_iters,args.test_batch_size,args.evaluator_seed)
    t = time.time()

    searcher.search()

    print('total searching time = {:.2f} hours'.format(
        (time.time() - t) / 3600))
    # print(args.save_dir)
    # set_dataprovider(args)

    # Seq2SeqSearch(args, torch.cuda.device_count())


if __name__ == '__main__':
    try:
        main()
        os._exit(0)
    except:
        import traceback
        traceback.print_exc()
        time.sleep(1)
        os._exit(1)
