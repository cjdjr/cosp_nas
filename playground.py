import torch.distributed as dist
import torch.multiprocessing as mp
import torch
import random
import numpy as np

from multiprocessing import Process

def f(rank):
    dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:23456', world_size=4, rank=rank)
    cost_1 = torch.rand(3,1)
    pi = torch.randint(4,(3,20)).float()
    info = torch.cat((cost_1,pi),dim=1)
    gather_info = [torch.ones_like(info) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_info, info)
    # print(rank, info)
    gather_info = torch.cat(tuple(gather_info),dim=0)
    print(gather_info.shape)
    if rank == 0:
        print("gather_info : ",gather_info)
        id = torch.argmin(gather_info[:,0]).item()
        print(id)
        print(tuple(map(int,gather_info[id][1:].numpy())))

def set_random_seed(seed):
    # Set the random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def eval():
    set_random_seed(1)
    print(random.random())
    print(random.random())

if __name__ == '__main__':
    set_random_seed(0)
    print(random.random())
    print(random.random())
    eval()
    # set_random_seed(1)
    # print(random.random())
    # print(random.random())
    # set_random_seed(0)
    print(random.random())
    print(random.random())
    eval()

