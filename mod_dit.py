import torch
import torch.nn as nn
import numpy as np
import h5py
import cv2
import os
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from diffusers.models import AutoencoderKL
from vae import decode_latent
from atari_dataset import AtariH5Dataset
from utils import timestep_embedding, unpatchify


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
        beta_1, gamma_1, alpha_1, beta_2, gamma_2, alpha_2 = self.adaLN_modulation(c).chunk(6, dim=1)
        
        x_norm = (1 + gamma_1.unsqueeze(1)) * self.norm1(x) + beta_1.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + alpha_1.unsqueeze(1) * attn_out
        
        x_norm = (1 + gamma_2.unsqueeze(1)) * self.norm2(x) + beta_2.unsqueeze(1)
        mlp_out = self.mlp(x_norm)
        x = x + alpha_2.unsqueeze(1) * mlp_out
        
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

        self.in_channels = in_channels
        self.context_frames = context_frames
        total_in_channels = in_channels + (in_channels * context_frames)

        self.x_embedder = nn.Conv2d(total_in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

        num_patches = (input_size // patch_size) ** 2

        # Conditioning
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, hidden_size) * 0.02, requires_grad=True)
        self.t_embedder = nn.Sequential(nn.Linear(256, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.act_embedder = nn.Embedding(num_actions, hidden_size)
        
        self.ctx_act_embedder = nn.Embedding(num_actions, hidden_size)
        self.ctx_act_proj = nn.Linear(hidden_size * context_frames, hidden_size)

        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)

    def forward(self, x, t, target_action, context_actions):
        t_emb = timestep_embedding(t, 256).to(x.device)
        t_vec = self.t_embedder(t_emb)
        
        act_vec = self.act_embedder(target_action)
        ctx_emb = self.ctx_act_embedder(context_actions)
        ctx_emb = ctx_emb.flatten(1)
        ctx_vec = self.ctx_act_proj(ctx_emb)
        c = t_vec + act_vec + ctx_vec
        
        x = self.x_embedder(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x, c)
            
        x = self.final_layer(x, c)
        return unpatchify(x, self.in_channels)


def train_dit_wm(dataset_path, epochs=10, batch_size=32, val_split=0.1, 
                 in_channels=4, context_frames=4, hidden_size=384, depth=6, num_heads=6, 
                 device='cuda'):    
    
    full_dataset = AtariH5Dataset(dataset_path, context_len=context_frames)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = DiT_WM(
        in_channels=in_channels, 
        context_frames=context_frames, 
        hidden_size=hidden_size, 
        depth=depth, 
        num_heads=num_heads, 
        num_actions=18
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    mse = nn.MSELoss()

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0
        update_interval = max(1, len(train_loader) // 10)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", miniters=update_interval)
        
        for target, context, tgt_act, ctx_acts in pbar:
            target, context = target.to(device), context.to(device)
            tgt_act, ctx_acts = tgt_act.to(device), ctx_acts.to(device)
            
            t = torch.randint(0, 1000, (target.shape[0],), device=device).long()
            noise = torch.randn_like(target)
            alpha_bar = 1 - (t / 1000.0).view(-1, 1, 1, 1)
            noisy_target = target * torch.sqrt(alpha_bar) + noise * torch.sqrt(1 - alpha_bar)
            
            model_input = torch.cat([noisy_target, context], dim=1)
            noise_pred = model(model_input, t, tgt_act, ctx_acts)
            
            loss = mse(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            if pbar.n % update_interval == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss_sum / len(train_loader)

        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for target, context, tgt_act, ctx_acts in val_loader:
                target, context = target.to(device), context.to(device)
                tgt_act, ctx_acts = tgt_act.to(device), ctx_acts.to(device)
                
                t = torch.randint(0, 1000, (target.shape[0],), device=device).long()
                noise = torch.randn_like(target)
                alpha_bar = 1 - (t / 1000.0).view(-1, 1, 1, 1)
                noisy_target = target * torch.sqrt(alpha_bar) + noise * torch.sqrt(1 - alpha_bar)
                
                model_input = torch.cat([noisy_target, context], dim=1)
                noise_pred = model(model_input, t, tgt_act, ctx_acts)
                val_loss_sum += mse(noise_pred, noise).item()
        avg_val_loss = val_loss_sum / len(val_loader)

        save_msg = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "dit_wm.pt")
            save_msg = "--> Saved Best Model!"

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} {save_msg}")

    print("Training Complete.")


