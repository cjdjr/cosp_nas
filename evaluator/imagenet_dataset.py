import os
import numpy as np
import torch
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

# train_dataset, valid_dataset = None, None
# train_bs, test_bs, train_it, test_it = None, None, None, None



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
    # global train_dataset,valid_dataset
    # global train_bs, test_bs, train_it, test_it

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

    set_value('train_dataset',train_dataset)
    set_value('valid_dataset',valid_dataset)
    set_value('train_batch_size',train_batch_size)
    set_value('test_batch_size',val_batch_size)
    set_value('max_train_iters',train_iters)
    set_value('max_test_iters',val_iters)

    # train_bs, test_bs, train_it, test_it = train_batch_size, val_batch_size, train_iters, val_iters

    # print("test ",train_bs, test_bs, train_it, test_it)

# def get_info():
#     global train_bs, test_bs, train_it, test_it
#     # print("before ",train_bs, test_bs, train_it, test_it)
#     train_batch_size, test_batch_size, max_train_iters, max_test_iters = train_bs, test_bs, train_it, test_it
#     print("test test ",train_batch_size, test_batch_size, max_train_iters, max_test_iters)
#     return train_batch_size, test_batch_size, max_train_iters, max_test_iters


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)


    print(len(train_dataset))
    print(train_dataset[np.random.randint(len(train_dataset))])


    print(len(valid_dataset))
    print(valid_dataset[np.random.randint(len(valid_dataset))])

    use_gpu = False
    train_batch_size = 128
    valid_batch_size = 200

    train_dataprovider = get_train_dataprovider(train_batch_size, use_gpu=use_gpu, num_workers=3)
    val_dataprovider = get_val_dataprovider(valid_batch_size, use_gpu=use_gpu, num_workers=2)

    train_data = train_dataprovider.next()
    val_data = val_dataprovider.next()

    print(train_data[0].mean().item())
    print(val_data[0].mean().item())

    from IPython import embed
    embed()

if __name__ == '__main__':
    main()
