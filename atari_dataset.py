# import h5py
# import torch
# from torch.utils.data import Dataset

# class AtariH5Dataset(Dataset):
#     def __init__(self, h5_path, context_len=4):
#         self.f = h5py.File(h5_path, 'r')
#         self.observations = self.f['observations']
#         self.actions = self.f['actions']
#         self.rewards = self.f['rewards']
#         self.terminated = self.f['terminated']

#         self.length = len(self.observations)
#         self.context_len = context_len

#     def __len__(self):
#         return self.length - self.context_len

#     def __getitem__(self, idx):
#         end_idx = idx + self.context_len
#         raw_frames = self.observations[idx:end_idx+1]
        
#         if raw_frames.dtype == 'uint8':
#             frames = torch.tensor(raw_frames).float() / 127.5 - 1.0
#         else:
#             frames = torch.tensor(raw_frames)

#         actions = torch.tensor(self.actions[idx:end_idx+1])
#         reward = torch.tensor(self.rewards[end_idx]).float()
#         raw_rewards = torch.tensor(self.rewards[idx:end_idx+1]).float()
#         done = torch.tensor(self.terminated[end_idx]).float()
#         raw_dones = torch.tensor(self.terminated[idx:end_idx+1]).float()
        
#         target_obs = frames[-1]
#         target_action = actions[-1].long()

#         h, w = frames.shape[-2], frames.shape[-1]
#         context_obs = frames[:-1].reshape(-1, h, w)
#         context_actions = actions[:-1].long()
#         context_rewards = raw_rewards[:-1].unsqueeze(-1)
#         context_dones = raw_dones[:-1].long()
        
#         return target_obs, context_obs, target_action, context_actions, reward, done, context_rewards, context_dones

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class AtariH5Dataset(Dataset):
    def __init__(self, h5_path, context_len=4):
        self.f = h5py.File(h5_path, 'r')
        self.observations = self.f['observations']
        self.actions = self.f['actions']
        self.rewards = self.f['rewards']
        self.terminated = self.f['terminated']
        self.context_len = context_len
        
        # Pre-calculate valid indices
        # A sequence is valid if no termination occurs in the context (first L frames)
        # The target frame (L+1) can be a terminal state, but we cannot cross a boundary.
        term_flags = np.array(self.terminated)
        total_len = len(self.observations)
        
        # Create a sliding window view of terminations to check for any True values
        # We need seq [i : i+context_len] to have NO True values.
        # This implementation is memory efficient for large datasets.
        
        valid_mask = np.ones(total_len - context_len, dtype=bool)
        for i in range(context_len):
            # If terminated[idx + i] is True, then any sequence starting at idx is invalid
            # because the context would contain a break.
            valid_mask[term_flags[i : total_len - context_len + i]] = False

        self.valid_indices = np.where(valid_mask)[0]
        self.length = len(self.valid_indices)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Map linear index to valid dataset index
        real_idx = self.valid_indices[idx]
        end_idx = real_idx + self.context_len
        
        # Slice data
        raw_frames = self.observations[real_idx : end_idx+1]
        
        if raw_frames.dtype == 'uint8':
            frames = torch.tensor(raw_frames).float() / 127.5 - 1.0
        else:
            frames = torch.tensor(raw_frames)

        actions = torch.tensor(self.actions[real_idx : end_idx+1])
        raw_rewards = torch.tensor(self.rewards[real_idx : end_idx+1]).float()
        raw_dones = torch.tensor(self.terminated[real_idx : end_idx+1]).float()
        
        # Targets (The frame we want to predict)
        target_obs = frames[-1]
        target_action = actions[-1].long()
        reward = raw_rewards[-1]
        done = raw_dones[-1]

        # Context (The history we condition on)
        h, w = frames.shape[-2], frames.shape[-1]
        context_obs = frames[:-1].reshape(-1, h, w)
        context_actions = actions[:-1].long()
        context_rewards = raw_rewards[:-1].unsqueeze(-1)
        context_dones = raw_dones[:-1].long()
        
        return target_obs, context_obs, target_action, context_actions, reward, done, context_rewards, context_dones