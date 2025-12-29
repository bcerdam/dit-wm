import h5py
import torch
from torch.utils.data import Dataset

class AtariH5Dataset(Dataset):
    def __init__(self, h5_path, context_len=4):
        self.f = h5py.File(h5_path, 'r')
        self.latents = self.f['latents']
        self.actions = self.f['actions']
        self.terminated = self.f['terminated']
        self.length = len(self.latents)
        self.context_len = context_len

    def __len__(self):
        return self.length - self.context_len

    def __getitem__(self, idx):
        end_idx = idx + self.context_len

        frames = torch.tensor(self.latents[idx:end_idx+1])
        actions = torch.tensor(self.actions[idx:end_idx+1])
        
        target_latent = frames[-1]
        target_action = actions[-1].long()
        
        context_latents = frames[:-1].reshape(-1, 8, 8)
        context_actions = actions[:-1].long()
        
        return target_latent, context_latents, target_action, context_actions