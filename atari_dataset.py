import h5py
import torch
from torch.utils.data import Dataset

class AtariH5Dataset(Dataset):
    def __init__(self, h5_path, context_len=4):
        self.f = h5py.File(h5_path, 'r')
        self.observations = self.f['observations']
        self.actions = self.f['actions']
        self.rewards = self.f['rewards']
        self.terminated = self.f['terminated']

        self.length = len(self.observations)
        self.context_len = context_len

    def __len__(self):
        return self.length - self.context_len

    def __getitem__(self, idx):
        end_idx = idx + self.context_len
        raw_frames = self.observations[idx:end_idx+1]
        
        if raw_frames.dtype == 'uint8':
            frames = torch.tensor(raw_frames).float() / 127.5 - 1.0
        else:
            frames = torch.tensor(raw_frames)

        actions = torch.tensor(self.actions[idx:end_idx+1])
        reward = torch.tensor(self.rewards[end_idx]).float()
        raw_rewards = torch.tensor(self.rewards[idx:end_idx+1]).float()
        done = torch.tensor(self.terminated[end_idx]).float()
        raw_dones = torch.tensor(self.terminated[idx:end_idx+1]).float()
        
        target_obs = frames[-1]
        target_action = actions[-1].long()

        h, w = frames.shape[-2], frames.shape[-1]
        context_obs = frames[:-1].reshape(-1, h, w)
        context_actions = actions[:-1].long()
        context_rewards = raw_rewards[:-1].unsqueeze(-1)
        context_dones = raw_dones[:-1].long()
        
        return target_obs, context_obs, target_action, context_actions, reward, done, context_rewards, context_dones