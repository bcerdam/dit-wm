import torch
import numpy as np
from tqdm import tqdm
from vae_dataset import VAEDataset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import lpips


class VAE_encoder(nn.Module):
    def __init__(self, latent_channel_dim=4):
        super().__init__()

        self.downscale_features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.z_mean = nn.Conv2d(in_channels=256, out_channels=latent_channel_dim, kernel_size=3, padding=1)
        self.z_log_var = nn.Conv2d(in_channels=256, out_channels=latent_channel_dim, kernel_size=3, padding=1)


    def forward(self, observations):
        downscale_features = self.downscale_features(observations)
        z_mean = self.z_mean(downscale_features)
        z_log_var = self.z_log_var(downscale_features)
        return z_mean, z_log_var


class VAE_decoder(nn.Module):
    def __init__(self, latent_channel_dim=4):
        super().__init__()

        self.upscale_features = nn.Sequential(
            nn.Conv2d(in_channels=latent_channel_dim, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        

    def forward(self, latent_observations):
        upscale_features = self.upscale_features(latent_observations)
        return upscale_features
        

class VAE(nn.Module):
    def __init__(self, latent_channel_dim, latent_spatial_dim, observation_resolution):
        super().__init__()

        self.latent_channel_dim = latent_channel_dim
        self.latent_spatial_dim = latent_spatial_dim
        self.observation_resolution = observation_resolution
        self.latent_dim = (self.latent_channel_dim, self.latent_spatial_dim, self.latent_spatial_dim)
        self.input_dim = (3, self.observation_resolution, self.observation_resolution)

        self.encoder = VAE_encoder()
        self.decoder = VAE_decoder()


    def sampler(self, z_mean, z_log_var):
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5*z_log_var) * epsilon


    def forward(self, observations):
        z_mean, z_log_var = self.encoder.forward(observations=observations)
        reconstructions = self.decoder.forward(latent_observations=self.sampler(z_mean=z_mean, z_log_var=z_log_var))
        return reconstructions, z_mean, z_log_var


def kl_loss(z_mean, z_log_var):
    kl = -0.5*(1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
    return torch.mean(torch.sum(kl, dim=[1, 2, 3]))


def train_vae(observations, val_split, batch_size, epochs, observation_resolution, latent_channel_dim, latent_spatial_dim, vae_weights_path, device='cuda'):
    observations = torch.tensor(np.vstack(observations), dtype=torch.uint8)
    dataset = VAEDataset(observations=observations)

    dataset_length = len(dataset)
    val_size = int(dataset_length * val_split)
    train_size = dataset_length - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = VAE(latent_channel_dim=latent_channel_dim,
                latent_spatial_dim=latent_spatial_dim,
                observation_resolution=observation_resolution).to(device)


    lpips_metric = lpips.LPIPS(net='alex').to(device)
    lpips_metric.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    reconstruction_loss = nn.MSELoss(reduction='sum')
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0
        update_interval = max(1, len(train_loader) // 10)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", miniters=update_interval)

        for observations in pbar:
            observations = observations.to(device)
            observations = (observations.float()/127.5)-1
            reconstructions, z_mean, z_log_var = model(observations)

            perceptual_loss = torch.sum(lpips_metric(reconstructions, observations))

            total_loss = reconstruction_loss(reconstructions, observations) + (10**-6)*kl_loss(z_mean=z_mean, z_log_var=z_log_var) + perceptual_loss*0.2

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss_sum += total_loss.item()
            if pbar.n % update_interval == 0:
                pbar.set_postfix(img=f"{total_loss.item():.2f}")

        avg_train_loss = train_loss_sum / len(train_loader)

        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for observations in val_loader:
                observations = observations.to(device)
                observations = (observations.float()/127.5)-1
                reconstructions, z_mean, z_log_var = model(observations)

                perceptual_loss = torch.sum(lpips_metric(reconstructions, observations))

                total_loss = reconstruction_loss(reconstructions, observations) + (10**-6)*kl_loss(z_mean=z_mean, z_log_var=z_log_var) + perceptual_loss*0.2

                val_loss_sum += total_loss.item()

        avg_val_loss = val_loss_sum / len(val_loader) if len(val_loader) > 0 else 0
        save_msg = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), vae_weights_path)
            save_msg = "--> Saved Best Model!"

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} {save_msg}")

    print("Training Complete.")
