import numpy as np
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data, lookback=60, target_col="return"):
        self.data = data.values
        self.lookback = lookback
        self.target_idx = data.columns.get_loc(target_col)

    def __len__(self):
        return len(self.data) - self.lookback

    def __getitem__(self, idx):
        X = self.data[idx:idx+self.lookback, :]
        y = self.data[idx+self.lookback, self.target_idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
