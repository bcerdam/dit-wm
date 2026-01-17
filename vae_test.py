import torch
import numpy as np
from vae_dataset import VAEDataset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, latent_channel_dim, latent_spatial_dim, observation_resolution):
        super().__init__()

        ### encoder ###
        self.latent_channel_dim = latent_channel_dim
        self.latent_spatial_dim = latent_spatial_dim
        self.latent_dim = (latent_channel_dim, latent_spatial_dim, latent_spatial_dim)
        self.input_dim = (3, observation_resolution, observation_resolution)

        self.downscale_features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1), 
                                                nn.ReLU(),
                                                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                                                nn.ReLU())
        
        with torch.no_grad():
            self.dummy = torch.zeros(1, *self.input_dim)
            self.dummy_features = self.downscale_features(self.dummy)
            self.flatten_dim = self.dummy_features.view(1, -1).shape[1]
            
        self.linear_1 = nn.Linear(in_features=self.flatten_dim, out_features=16)

        ### Aqui hay que arreglar out_features, basicamente deberia ser self.latent_dim flattened, para despues
        ### poder hacerle reshape a la dimension del espacio latente.

        self.z_mean = nn.Linear(in_features=16, out_features=self.latent_dim.view(1, -1))
        self.z_log_var = nn.Linear(in_features=16, out_features=self.latent_dim.view(1, -1))
        ### encoder ###


        ### decoder ###
        self.linear_1 = nn.Linear(in_features=self.latent_dim.view(1, -1), out_features=self.flatten_dim)
        # self.upscale_features = nn.Sequential(nn.ConvTranspose2d(in_channels=self.flatten_dim.view()))




def train_vae(observations, val_split, batch_size, epochs, observation_resolution, latent_channel_dim, latent_spatial_dim):
    observations = torch.tensor(np.vstack(observations))
    dataset = VAEDataset(observations=observations)

    dataset_length = len(dataset)
    val_size = int(dataset_length * val_split)
    train_size = dataset_length - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = VAE(latent_channel_dim=latent_channel_dim,
                latent_spatial_dim=latent_spatial_dim,
                observation_resolution=observation_resolution)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # best_val_loss = float('inf')
    # for epoch in range(epochs):
    #     pass

    return 0