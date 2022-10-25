import pandas as pd
import torch
import numpy as np
import scipy.sparse
from torch.utils.data import Dataset
import random

random.seed(42)

class CustomDataset(Dataset):

    def __init__(self, fname) -> None:
        """
        1. construct a sparse matrix by coo_matrix format
        2. user pd.factorize to get the user & item tokens

        note: there are two kinds of index. one for nn.embedding and one for sparse matrix index
        user and item could be mapped into the same token, e.g. 42, but they will have different embeddings
        for items, idx_mat = idx_emb + num_user
        """
        super().__init__()
        self.df = pd.read_csv(fname, header=None)
        u_codes, u_uniques = pd.factorize(self.df[0])
        i_codes, i_uniques = pd.factorize(self.df[1])
        self.df[0], self.df[1] = u_codes, i_codes
        self.num_user = len(u_uniques)
        self.num_item = len(i_uniques)
        num_node = self.num_user + self.num_item
        ones = np.ones_like(u_codes)
        matrix = scipy.sparse.coo_matrix((ones, (u_codes, i_codes+self.num_user)), shape=(num_node, num_node))
        self.dist_matrix = scipy.sparse.csgraph.shortest_path(matrix, directed=False)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index: int):
        return self.df[0][index], self.df[1][index]
    
    def split(self):
        # train: val: test = 0.8 : 0.1 : 0.1
        train_base = self.df.groupby(0).sample()
        num_train_left = int(0.8 * len(self.df)) - len(train_base)
        num_val = int(0.1 * len(self.df))
        num_test = len(self.df) - int(0.8 * len(self.df)) - num_val
        train_idx = set(train_base.index)
        idx_pool = set(range(len(self.df))) - train_idx
        train_left_idx = set(random.sample(idx_pool, num_train_left))
        idx_pool -= train_left_idx
        train_idx = train_left_idx | train_idx
        val_idx = set(random.sample(idx_pool, num_val))
        idx_pool -= val_idx
        assert int(0.8 * len(self.df)) == len(train_idx)
        assert num_val == len(val_idx)
        assert num_test == len(idx_pool)
        train_dataset = torch.utils.data.Subset(self, list(train_idx))
        val_dataset = torch.utils.data.Subset(self, list(val_idx))
        test_dataset = torch.utils.data.Subset(self, list(idx_pool))
        return train_dataset, val_dataset, test_dataset



def test():
    fname = '~/cdr/data/dataset/5_core_ratings_Musical_Instruments.csv'
    abc = CustomDataset(fname)


if __name__ == '__main__':
    test()