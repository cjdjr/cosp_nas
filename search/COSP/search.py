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
from tqdm import tqdm
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

import functools
print = functools.partial(print, flush=True)

from cosp_nas.utils import torch_load_cpu, get_inner_model, set_random_seed, get_logger, move_to, clip_grad_norms, log_values

from .network import AttentionModel, set_decode_type
from .reinforce_baselines import NoBaseline, WarmupBaseline, ExponentialBaseline

from .subnet_dataset import SubnetDataset


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
        model = model.module
    model.set_decode_type(decode_type)

class COSPSearcher(object):

    def __init__(self, args, supernet):

        self.args = args
        self.logger = get_logger(os.path.join(self.args.save_dir, 'process.log'))
        # supernet
        self.supernet = supernet

    def check_tblogger(self):
        self.tb_logger = None
        if not self.args.no_tensorboard:
            self.tb_logger = TbLogger(os.path.join(self.args.save_dir))
            # Pretty print the run args
        pp.pprint(vars(self.args))

    def validate(self, model, log=False):
        # Validate
        self.logger.info('Validating...')

        # Fix the process of sample in validate 

        set_random_seed( self.args.eval_seed )

        input = SubnetDataset(num_samples=1,test=True)[0][None,:,:].expand(self.args.val_sample, self.args.n_layer * self.args.n_op,2).to(self.args.device)
        # print("val dataset input : ",input[0])
        with torch.no_grad():
            model.eval()
            set_decode_type(model, "sampling")
            cost_1, cost_5, _ , pi = model(input, return_pi=True)
            cost_1 = cost_1[:,None]
            # cost_1_min = cost_1.min()
            pi = pi.float()
            info = torch.cat((cost_1,pi),dim=1)
            cost_1_sum = cost_1.sum()
            cost_5_min = cost_5.min()
            cost_5_sum = cost_5.sum()

        # cost_1_min = gather_min(cost_1_min)
        gather_info = [torch.ones_like(info) for _ in range(dist.get_world_size())]
        dist.all_gather(gather_info, info)
        gather_info = torch.cat(tuple(gather_info),dim=0)
        cost_1_min_id = torch.argmin(gather_info[:,0]).item()
        cost_1_min = gather_info[cost_1_min_id][0].item()
        cand = tuple([x%4 for x in map(int,gather_info[cost_1_min_id][1:].cpu().numpy())])
        cost_1_mean = gather_sum(cost_1_sum)/args.val_sample
        cost_5_min = gather_min(cost_5_min)
        cost_5_mean = gather_sum(cost_5_sum)/args.val_sample
        if args.local_rank == 0:
            print("cost_1_min = {} cost_1_mean = {} cost_5_min={} cost_5_mean={} \n cand : {}".format(cost_1_min,cost_1_mean,cost_5_min,cost_5_mean,cand))
            if log:
                filename, _ = os.path.splitext(args.eval_model)
                filename = filename + ".log"
                with open(filename,'w') as f:
                    f.write("cost_1_min = {} cost_1_mean = {} cost_5_min={} cost_5_mean={} \n cand : {}".format(cost_1_min,cost_1_mean,cost_5_min,cost_5_mean,cand))

        # cannot make the seed same in all epoch
        self.args.seed += 1            
        set_random_seed( self.args.seed )

        return cost_1_min,cost_1_mean,cost_5_min,cost_5_mean
    
    def train_batch(
            self,
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
    ):

        # x, bl_val = baseline.unwrap_batch(batch)

        x = batch

        x = move_to(x, self.args.device)

        # bl_val = move_to(bl_val, self.args.device) if bl_val is not None else None
        # if batch_id == 0:
        #     print(args.device," : ",batch_id," : ",x[0])
        # return 
        # Evaluate model, get costs and log probabilities

        # cost_1, cost_5, log_likelihood= model(x)
        cost_1 = torch.tensor([0.])
        cost_5 = torch.tensor([0.])
        log_likelihood = torch.tensor([0.])

        # global info
        # for i in range(cost_1.size(0)):
        #     info[pi[i].item()]={'err':(cost_1[i].item(),cost_5[i].item())}

        # Evaluate baseline, get baseline loss if any (only for critic)
        # bl_val, bl_loss = baseline.eval(x, cost_1) if bl_val is None else (bl_val, 0)

        # Calculate loss
        reinforce_loss = (cost_1 * log_likelihood).mean()
        loss = reinforce_loss 

        # Perform backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        # Clip gradient norms and get (clipped) gradient norms for logging
        grad_norms = clip_grad_norms(optimizer.param_groups, self.args.max_grad_norm)
        optimizer.step()

        # gather 

        cost_1 = cost_1.mean()
        log_likelihood = log_likelihood.mean()
        reinforce_loss = reinforce_loss
        # bl_loss = gather_mean(bl_loss)
        # Logging

        if step % int(self.args.log_step) == 0:
            log_values(cost_1, grad_norms, epoch, batch_id, step,
                    log_likelihood, reinforce_loss, None, self.tb_logger, self.args)

    def train_epoch(self, model, optimizer, baseline, lr_scheduler, epoch):

        self.logger.info("Train!")
        self.logger.info("Start train epoch {}, lr={} for run {} , seed={}".format(epoch, optimizer.param_groups[0]['lr'], self.args.run_name, self.args.seed))

        step = epoch * (self.args.epoch_size // self.args.batch_size)

        start_time = time.time()

        if not self.args.no_tensorboard:
            self.tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

        # Generate new training data for each epoch
        training_dataset = baseline.wrap_dataset(SubnetDataset(num_samples=self.args.epoch_size))
        training_dataloader = DataLoader(training_dataset, batch_size=self.args.batch_size, num_workers=0)

        # Put model in train mode!
        model.train()
        set_decode_type(model, "sampling")

        for batch_id, batch in enumerate(tqdm(training_dataloader, disable=self.args.no_progress_bar)):

            self.train_batch(
                model,
                optimizer,
                baseline,
                epoch,
                batch_id,
                step,
                batch
            )
            step += 1

        epoch_duration = time.time() - start_time

        # print(epoch_duration.shape)

        self.logger.info("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

        self.logger.info('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict(),
            },
            os.path.join(self.args.save_dir, 'epoch-{}.pt'.format(epoch))
        )

        self.args.eval_model = os.path.join(args.save_dir, 'epoch-{}.pt'.format(epoch))

        # cost_1_min,cost_1_mean,cost_5_min,cost_5_mean = validate(model, True)

        # if not self.args.no_tensorboard:
        #     tb_logger.log_value('cost_1_min', cost_1_min, step)
        #     tb_logger.log_value('cost_1_mean', cost_1_mean, step)
        #     tb_logger.log_value('cost_5_min', cost_5_min, step)
        #     tb_logger.log_value('cost_5_mean', cost_5_mean, step)

        baseline.epoch_callback(model, epoch)

        # lr_scheduler should be called at end of epoch
        lr_scheduler.step()

    def search(self):

        self.check_tblogger()

        with open(os.path.join(self.args.save_dir, "args.json"), 'w') as f:
            json.dump(vars(self.args), f, indent=True)
        
        # Load data from load_path
        load_data = {}
        assert self.args.load_path is None or self.args.resume is None, "Only one of load path and resume can be given"
        load_path = self.args.load_path if self.args.load_path is not None else self.args.resume
        if load_path is not None:
            print('  [*] Loading data from {}'.format(load_path))
            load_data = torch_load_cpu(load_path)
        
        # Set the random seed
        set_random_seed(self.args.seed)

        # Set the device 
        if torch.cuda.is_available():
            self.args.device = torch.device('cuda')
        else:
            self.args.device = torch.device('cpu')

        model = AttentionModel(
        self.args.embedding_dim,
        self.args.hidden_dim,
        self.supernet,
        n_encode_layers=self.args.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=self.args.normalization,
        tanh_clipping=self.args.tanh_clipping,
        checkpoint_encoder=self.args.checkpoint_encoder,
        shrink_size=self.args.shrink_size
        ).to(self.args.device)
        model = torch.nn.DataParallel(model).cuda()

        model_ = get_inner_model(model)
        model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

        # Initialize baseline
        if self.args.baseline == 'exponential':
            baseline = ExponentialBaseline(self.args.exp_beta)
        else:
            assert self.args.baseline is None, "Unknown baseline: {}".format(self.args.baseline)
            baseline = NoBaseline()

        if self.args.bl_warmup_epochs > 0:
            baseline = WarmupBaseline(baseline, self.args.bl_warmup_epochs, warmup_exp_beta=self.args.exp_beta)

        # Load baseline from data, make sure script is called with same type of baseline
        if 'baseline' in load_data:
            baseline.load_state_dict(load_data['baseline'])
        
        # Initialize optimizer
        optimizer = optim.Adam(
            [{'params': model.parameters(), 'lr': self.args.lr_model}]
            + (
                [{'params': baseline.get_learnable_parameters(), 'lr': self.args.lr_critic}]
                if len(baseline.get_learnable_parameters()) > 0
                else []
            )
        )

        # Load optimizer state
        if 'optimizer' in load_data:
            optimizer.load_state_dict(load_data['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    # if isinstance(v, torch.Tensor):
                    if torch.is_tensor(v):
                        state[k] = v.to(self.args.device)

        # Initialize learning rate scheduler, decay by lr_decay once per epoch!
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: self.args.lr_decay ** epoch)

        if self.args.resume:
            epoch_resume = int(os.path.splitext(os.path.split(self.args.resume)[-1])[0].split("-")[1])

            torch.set_rng_state(load_data['rng_state'])
            if self.args.use_cuda:
                torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
            # Set the random states
            # Dumping of state was done before epoch callback, so do that now (model is loaded)
            baseline.epoch_callback(model, epoch_resume)
            print("Resuming after {}".format(epoch_resume))
            self.args.epoch_start = epoch_resume + 1
        
        print("build finish ! ")
    

        if self.args.eval_only:
            self.args.eval_model = self.args.load_path
            validate(model, True)
        else:
            for epoch in range(self.args.epoch_start, self.args.epoch_start + self.args.n_epochs):
                self.train_epoch(
                    model,
                    optimizer,
                    baseline,
                    lr_scheduler,
                    epoch,
                )

