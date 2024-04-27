import torch
import numpy as np

from torch.utils.data import Dataset


class DiamondsPytorchDataset(Dataset):
    def __init__(self, data, cat_features, num_features, label=None):
        self.data = data
        self.cat_features = cat_features
        self.num_features = num_features
        self.label = label

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        x_num = torch.from_numpy(
            row[self.num_features].to_numpy().astype(np.float32)
        ).view(-1)

        x_cat = torch.from_numpy(
            row[self.cat_features].to_numpy().astype(np.int32)
        ).view(-1)
        
        if self.label:
            target = torch.from_numpy(row[[self.label]].to_numpy().astype(np.float32))
            return x_cat, x_num, target
        return x_cat, x_num
