from torch.utils.data import Dataset
import torch
import os
import pickle
import numpy as np

class SubnetDataset(Dataset):
    
    def __init__(self, num_samples, filename=None, n_layer=20, n_op=4, test=False):
        super(SubnetDataset, self).__init__()

        def get_data(n_layer, n_op):
            ans=torch.Tensor([])
            for layer in range(n_layer):
                ans = torch.cat((ans,torch.stack((torch.Tensor([layer]).expand(n_op),torch.Tensor(np.random.choice(n_op, n_op))),1)),dim=0)
            return ans

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in data]
        else:
            if test:
                self.data = [torch.cat(tuple([torch.FloatTensor([i,j])[None,:] for i in range(n_layer) for j in range(n_op)]),dim=0)]
            else:    
                self.data = [get_data(n_layer, n_op) for i in range(num_samples)] 
        # print(self.data[0])


        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]