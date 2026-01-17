import torch
from torch.utils.data import Dataset


class VAEDataset(Dataset):
    def __init__(self, observations):
        self.observations = observations
        self.length = len(self.observations)
        pass


    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        return torch.permute(self.observations[idx], (2,0 , 1))