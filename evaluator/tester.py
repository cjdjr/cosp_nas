import torch

from .imagenet_dataset import get_train_dataprovider, get_val_dataprovider
import tqdm

assert torch.cuda.is_available()

train_batch_size, test_batch_size, max_train_iters, max_test_iters = None, None, None, None

def evaluator_setting(a,b,c,d):
    global train_batch_size, test_batch_size, max_train_iters, max_test_iters
    train_batch_size, test_batch_size, max_train_iters, max_test_iters = a, b, c, d

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
    global train_batch_size, test_batch_size, max_train_iters, max_test_iters
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
