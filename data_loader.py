import torch
import numpy as np 
from torch.utils.data import Dataset, DataLoader

class SequenceForecastDataset(Dataset):

    def __init__(self, data, seq_len, pred_horizon):
        """
        data: tensor or ndarray shaped [traj, time, state_dim]
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        self.data = data
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        
        self.num_traj, self.T, self.state_dim = data.shape
        
        # each trajectory gives (T - seq_len - pred_horizon + 1) samples
        self.samples_per_traj = self.T - seq_len - pred_horizon + 1

    def __len__(self):
        return self.num_traj * self.samples_per_traj

    def __getitem__(self, idx):
        traj_idx = idx // self.samples_per_traj
        start_t = idx % self.samples_per_traj
        
        X = self.data[traj_idx, start_t : start_t + self.seq_len]
        Y = self.data[traj_idx, 
                      start_t + self.seq_len : 
                      start_t + self.seq_len + self.pred_horizon]
        
        return X, Y