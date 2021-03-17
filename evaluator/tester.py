import torch
assert torch.cuda.is_available()
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import cv2
import tarfile
import PIL
from PIL import Image
import tqdm
from cosp_nas.utils import set_random_seed

class OpencvResize(object):

    def __init__(self, size=256):
        self.size = size

    def __call__(self, img):
        assert isinstance(img, PIL.Image.Image)
        img = np.asarray(img) # (H,W,3) RGB
        img = img[:,:, ::-1] # 2 BGR
        img = np.ascontiguousarray(img)
        H, W, _ = img.shape
        target_size = (int(self.size/H * W + 0.5), self.size) if H < W else (self.size, int(self.size/W * H + 0.5))
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        img = img[:,:, ::-1] # 2 RGB
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)
        return img

class ToBGRTensor(object):

    def __call__(self, img):
        assert isinstance(img, (np.ndarray, PIL.Image.Image))
        if isinstance(img, PIL.Image.Image):
            img = np.asarray(img)
        img = img[:,:, ::-1] # 2 BGR
        img = np.transpose(img, [2, 0, 1]) # 2 (3, H, W)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        return img

class DataIterator(object):

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data[0], data[1]


train_dir = os.getcwd()+'/data/train'
val_dir = os.getcwd()+'/data/val'

assert os.path.exists(train_dir)
assert os.path.exists(val_dir)

print("##########")
train_dataset, valid_dataset = None, None
train_bs, test_bs, train_it, test_it = None, None, None, None



def get_train_dataprovider(batch_size, *, num_workers, use_gpu):
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_gpu)
    train_dataprovider = DataIterator(train_loader)
    return train_dataprovider

def get_val_dataprovider(batch_size, *, num_workers, use_gpu):
    val_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_gpu
    )
    val_dataprovider = DataIterator(val_loader)
    return val_dataprovider

def init_evaluator(train_iters, train_batch_size, val_iters, val_batch_size, evaluator_seed = 0):

    global train_dataset,valid_dataset
    global train_bs, test_bs, train_it, test_it

    # print(train_iters, train_batch_size, val_iters, val_batch_size, evaluator_seed)

    # fix the evaluator
    set_random_seed(evaluator_seed)

    train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(0.5),
        ToBGRTensor(),
    ])
    )
    valid_dataset = datasets.ImageFolder(
        val_dir,
        transforms.Compose([
            OpencvResize(256),
            transforms.CenterCrop(224),
            ToBGRTensor(),
        ])
    )
    train_dataset = torch.utils.data.random_split(train_dataset,[train_iters*train_batch_size,len(train_dataset)-train_iters*train_batch_size])[0]
    valid_dataset = torch.utils.data.random_split(valid_dataset,[val_iters*val_batch_size,len(valid_dataset)-val_iters*val_batch_size])[0]

    train_bs, test_bs, train_it, test_it = train_batch_size, val_batch_size, train_iters, val_iters

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def no_grad_wrapper(func):
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return new_func


@no_grad_wrapper
def get_cand_err(model, cand):

    train_batch_size, test_batch_size, max_train_iters, max_test_iters = train_bs, test_bs, train_it, test_it
    
    # print(train_batch_size, test_batch_size, max_train_iters, max_test_iters)
    # return 0.,0.

    # from IPython import embed
    # embed()
    assert train_batch_size != None, "Please set the evaluator first of all !"

    use_gpu = True
    train_dataprovider = get_train_dataprovider(
        train_batch_size, use_gpu=use_gpu, num_workers=8)
    val_dataprovider = get_val_dataprovider(
        test_batch_size, use_gpu=use_gpu, num_workers=8)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')



    print('clear bn statics....')
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean = torch.zeros_like(m.running_mean)
            m.running_var = torch.ones_like(m.running_var)

    print('train bn with training set (BN sanitize) ....')
    model.train()

    for step in tqdm.tqdm(range(max_train_iters)):
    # for step in range(max_train_iters):
        # print('train step: {} total: {}'.format(step,max_train_iters))
        data, target = train_dataprovider.next()
        # print('get data',data.shape)

        target = target.type(torch.LongTensor)

        data, target = data.to(device), target.to(device)

        output = model(data, cand)

        del data, target, output

    top1 = 0
    top5 = 0
    total = 0

    print('starting test....')
    model.eval()

    for step in tqdm.tqdm(range(max_test_iters)):
    # for step in range(max_test_iters):
        # print('test step: {} total: {}'.format(step,max_test_iters))
        data, target = val_dataprovider.next()
        batchsize = data.shape[0]
        # print('get data',data.shape)
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)

        logits = model(data, cand)

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))

        # print(prec1.item(),prec5.item())

        top1 += prec1.item() * batchsize
        top5 += prec5.item() * batchsize
        total += batchsize

        del data, target, logits, prec1, prec5

    top1, top5 = top1 / total, top5 / total

    top1, top5 = 1 - top1 / 100, 1 - top5 / 100

    print('top1: {:.2f} top5: {:.2f}'.format(top1 * 100, top5 * 100))

    return top1, top5

def get_costs(model, input, pi):
    """
    :param model: supernet
    :param input: (batch_size, graph_size, node_dim) 
    :param pi: (batch_size)
    :return:
    """
    cost_1 = torch.zeros(input.size(0),device=input.device)
    cost_5 = torch.zeros(input.size(0),device=input.device)
    # return cost_1,cost_5, None
    # print("input 0 ",input[0].shape)
    # print("pi 0 ",pi[0])
    for i in range(input.size(0)):
        cand = list(map(int,input[i][pi[i]][:,1].cpu().tolist()))
        # print("cand = {} ".format(cand))
        cost_1[i] , cost_5[i] = get_cand_err(model,cand)

    return cost_1, cost_5, None
    
def main():
    pass
