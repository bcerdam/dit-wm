import torch
import torch.nn as nn
import numpy as np
import h5py
import cv2
import os
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from diffusers.models import AutoencoderKL
from vae import decode_latent  # Ensure vae.py is in the same folder


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # Maybe it would be better to call them gamma_1, beta_1, alpha_1, gamma_2, beta_2, gamma_2
        # Are they learned with an mlp?
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Attention
        x_norm = (1 + scale_msa.unsqueeze(1)) * self.norm1(x) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP
        x_norm = (1 + scale_mlp.unsqueeze(1)) * self.norm2(x) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = (1 + scale.unsqueeze(1)) * self.norm_final(x) + shift.unsqueeze(1)
        return self.linear(x)

class DiT_WM(nn.Module):
    def __init__(self, input_size=8, patch_size=2, in_channels=4, context_frames=4, 
                 hidden_size=384, depth=6, num_heads=6, num_actions=18):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.context_channels = in_channels * context_frames
        
        # Input Embedding: Accepts Noisy Latent (4) + Context Stack (16) = 20 channels
        total_in_channels = in_channels + self.context_channels
        self.x_embedder = nn.Conv2d(total_in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        
        # Positional Embedding
        num_patches = (input_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # Condition Embeddings (Timestep + Action)
        self.t_embedder = nn.Sequential(nn.Linear(256, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.act_embedder = nn.Embedding(num_actions, hidden_size)

        # Blocks
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels) # Output is just noise (4ch)

    def forward(self, x, t, action):
        # x: (B, 20, 8, 8), t: (B,), action: (B,)
        
        # 1. Prepare Conditioning Vector (t + action)
        t_emb = timestep_embedding(t, 256).to(x.device)
        c = self.t_embedder(t_emb) + self.act_embedder(action)
        
        # 2. Patchify
        x = self.x_embedder(x).flatten(2).transpose(1, 2) # (B, N, D)
        x = x + self.pos_embed

        # 3. Transformer Blocks
        for block in self.blocks:
            x = block(x, c)
            
        # 4. Unpatchify
        x = self.final_layer(x, c)
        return unpatchify(x, self.in_channels)

# --- Utilities ---
def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-torch.log(torch.tensor(max_period)) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2: embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def unpatchify(x, channels):
    # (B, N, patch_size**2 * C) -> (B, C, H, W)
    p = 2 # hardcoded patch size match
    h = w = int(x.shape[1] ** 0.5)
    x = x.reshape(shape=(x.shape[0], h, w, p, p, channels))
    x = torch.einsum('nhwpqc->nchpwq', x)
    return x.reshape(shape=(x.shape[0], channels, h * p, h * p))

# --- Data & Training ---
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
        # We need indices [idx, idx+1, ..., idx+context_len]
        # Target is the LAST frame. Context is previous ones.
        end_idx = idx + self.context_len
        
        # Fetch window
        frames = torch.tensor(self.latents[idx:end_idx+1]) # (T+1, 4, 8, 8)
        
        # Action that led to the target frame (at end_idx) is at end_idx-1
        action = torch.tensor(self.actions[end_idx-1]).long()
        
        # Target: The frame we want to predict (noisy)
        target_latent = frames[-1]
        
        # Context: The history stack
        context_latents = frames[:-1].reshape(-1, 8, 8) # Stack channels: (4*4, 8, 8)
        
        return target_latent, context_latents, action


# --- Training Function ---
def train_dit_wm(dataset_path, epochs=10, batch_size=32, val_split=0.1, device='cuda'):
    print(f"Initializing DiT World Model Training on {device}...")
    
    full_dataset = AtariH5Dataset(dataset_path)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = DiT_WM(in_channels=4, context_frames=4, num_actions=18).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    mse = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0
        
        # 10% Update Interval
        update_interval = max(1, len(train_loader) // 10)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", miniters=update_interval)
        
        for target, context, action in pbar:
            target, context, action = target.to(device), context.to(device), action.to(device)
            
            t = torch.randint(0, 1000, (target.shape[0],), device=device).long()
            noise = torch.randn_like(target)
            
            # Linear alpha schedule: alpha_bar = 1 - t/1000
            alpha_bar = 1 - (t / 1000.0).view(-1, 1, 1, 1)
            noisy_target = target * torch.sqrt(alpha_bar) + noise * torch.sqrt(1 - alpha_bar)
            
            model_input = torch.cat([noisy_target, context], dim=1)
            noise_pred = model(model_input, t, action)
            
            loss = mse(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            
            # Manually update description only periodically to avoid clutter
            if pbar.n % update_interval == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss_sum / len(train_loader)

        # Validation
        model.eval()
        val_loss_sum = 0
        if len(val_loader) > 0:
            with torch.no_grad():
                for target, context, action in val_loader:
                    target, context, action = target.to(device), context.to(device), action.to(device)
                    t = torch.randint(0, 1000, (target.shape[0],), device=device).long()
                    noise = torch.randn_like(target)
                    alpha_bar = 1 - (t / 1000.0).view(-1, 1, 1, 1)
                    noisy_target = target * torch.sqrt(alpha_bar) + noise * torch.sqrt(1 - alpha_bar)
                    
                    model_input = torch.cat([noisy_target, context], dim=1)
                    noise_pred = model(model_input, t, action)
                    val_loss_sum += mse(noise_pred, noise).item()
            avg_val_loss = val_loss_sum / len(val_loader)
        else:
            avg_val_loss = 0.0
        
        # Save Best Model Logic
        save_msg = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "dit_wm.pt")
            save_msg = "--> Saved Best Model!"

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} {save_msg}")

    print("Training Complete.")


# --- Visualization Function ---
def visualize_denoising(model, weights_path, dataset_path, output_filename='denoising.mp4', device='cuda'):
    print(f"Creating denoising video from {weights_path}...")
    
    # 1. Load Model
    model.load_state_dict(torch.load(weights_path))
    model.to(device)
    model.eval()

    # 2. Load VAE for decoding (Required for video)
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    # 3. Get a random sample
    dataset = AtariH5Dataset(dataset_path)
    rand_idx = np.random.randint(0, len(dataset))
    target_clean, context, action = dataset[rand_idx]
    
    # Prepare batch of size 1
    target_clean = target_clean.unsqueeze(0).to(device) # (1, 4, 8, 8)
    context = context.unsqueeze(0).to(device)           # (1, 16, 8, 8)
    action = action.unsqueeze(0).to(device)             # (1,)

    # 4. Initialize Pure Noise
    xt = torch.randn_like(target_clean)
    
    # Setup Video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, 10.0, (64, 64)) # 10 fps

    print("Sampling reverse process...")
    num_steps = 1000
    
    with torch.no_grad():
        for t_idx in tqdm(reversed(range(num_steps)), total=num_steps):
            t = torch.tensor([t_idx], device=device).long()
            
            # Predict noise
            model_input = torch.cat([xt, context], dim=1)
            noise_pred = model(model_input, t, action)
            
            # DDPM Update Step (Simplified for our Linear Schedule)
            # Alpha bar schedule: 1 - t/1000
            alpha_bar_t = 1 - t_idx / 1000.0
            alpha_bar_prev = 1 - (t_idx - 1) / 1000.0 if t_idx > 0 else 1.0
            beta_t = 1 - (alpha_bar_t / alpha_bar_prev)
            
            if t_idx > 0:
                sigma_t = np.sqrt(beta_t)
                z = torch.randn_like(xt)
            else:
                sigma_t = 0
                z = 0
                
            # x_{t-1} calculation
            # This is the standard DDPM formula derived for our specific alpha_bar
            alpha_t = 1.0 - beta_t
            xt = (1 / np.sqrt(alpha_t)) * (xt - (beta_t / np.sqrt(1 - alpha_bar_t)) * noise_pred) + sigma_t * z

            # Capture frame every 20 steps (50 frames total) or last step
            if t_idx % 20 == 0 or t_idx == 0:
                # Decode
                frame_pixels = decode_latent(xt, vae, device=device)
                img_bgr = cv2.cvtColor(frame_pixels, cv2.COLOR_RGB2BGR)
                out.write(img_bgr)

    out.release()
    print(f"Video saved to {output_filename}")


model = DiT_WM(in_channels=4, context_frames=4, num_actions=18).to('cuda')
visualize_denoising(model, 'dit_wm.pt', 'atari_dataset.h5')