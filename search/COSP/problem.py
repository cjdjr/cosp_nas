import torch

class Subnet(object):
    def __init__(self, n_layer, n_op, input):
        super(Subnet, self).__init__()
        self.device = input.device
        # print("device = ",self.device)
        self.n_layer = n_layer
        self.n_op = n_op
        self.batch_size, self.node_num, _ = input.shape
        # print("batch_size={} node_num={}".format(self.batch_size,self.node_num))
        self.now = 0
        self.start = 0
        self.len = []
        self.current_node = prev_a = torch.zeros(self.batch_size, 1, dtype=torch.long,device = self.device)
        i=0
        while i<self.node_num:
            j=i
            while j+1<self.node_num and input[0][j+1][0]==input[0][i][0]:
                j+=1
            self.len.append(j-i+1)
            i=j+1
        assert(len(self.len)==self.n_layer)

    def update(self,selected):
        self.current_node=selected[:,None]  # Add dimension for step
        self.start+=self.len[self.now]
        self.now +=1

    def get_current_node(self):
        return self.current_node

    def get_mask(self):
        mask = torch.ones(self.node_num,device = self.device)
        # mask = torch.Tensor([1.,1.])
        # mask = torch.ones(2)
        # mask = mask.expand(self.batch_size,-1)
        # mask = torch.randn(self.node_num)
        mask[self.start:self.start+self.len[self.now]] = 0
        mask = mask>0
        # print("mask shape : {} , batch_size : {}".format(mask.shape,self.batch_size))
        mask = mask.expand(self.batch_size,-1)
        # while True:
        #     pass
        return mask[:,None,:]

def make_state(supernet,input):
    return Subnet(supernet.n_layer,supernet.n_op,input)
