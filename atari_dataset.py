import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class AtariH5Dataset(Dataset):
    def __init__(self, h5_path, context_length):
        self.f = h5py.File(h5_path, 'r')
        self.observations = self.f['observations']
        self.actions = self.f['actions']
        self.rewards = self.f['rewards']
        self.termination_status = self.f['termination_status']

        self.length = len(self.observations)
        self.context_length = context_length

        max_start_idx = self.length - self.context_length 
        valid_mask = np.ones(max_start_idx, dtype=bool)
        term_indices = np.where(self.termination_status[:] == True)[0]

        for t in term_indices:
            bad_start = max(0, t - self.context_length + 1)
            bad_end = min(max_start_idx, t + 1)
            if bad_start < bad_end:
                valid_mask[bad_start : bad_end] = False

        self.valid_indices = np.where(valid_mask)[0]
        self.length = len(self.valid_indices)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        valid_idx = self.valid_indices[idx]
        end_idx = valid_idx + self.context_length
        raw_observations = self.observations[valid_idx:end_idx+1]
        
        if raw_observations.dtype == 'uint8':
            raw_observations = torch.tensor(raw_observations).float() / 127.5 - 1.0
        else:
            raw_observations = torch.tensor(raw_observations)

        actions = torch.tensor(self.actions[valid_idx:end_idx+1])
        reward = torch.tensor(self.rewards[end_idx]).float()
        termination_status = torch.tensor(self.termination_status[end_idx]).float()
        
        target_obs = raw_observations[-1]
        target_action = actions[-1].long()

        h, w = raw_observations.shape[-2], raw_observations.shape[-1]
        context_obs = raw_observations[:-1].reshape(-1, h, w)
        context_actions = actions[:-1].long()
        
        return target_obs, context_obs, target_action, context_actions, reward, termination_status
