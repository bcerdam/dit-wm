import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from atari_dataset import AtariH5Dataset
from utils import unpatchify


class FourierEmbedding(nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        return torch.cat([x.cos(), x.sin()], dim=1)


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


class ModDiT(nn.Module):
    def __init__(self, num_actions, input_size=8, patch_size=2, in_channels=4, context_frames=4, 
                 hidden_size=384, depth=6, num_heads=6, sigma_data=0.5):
        super().__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.sigma_data = sigma_data
        self.context_frames = context_frames
        total_in_channels = in_channels + (in_channels * context_frames)
        num_patches = (input_size // patch_size) ** 2

        self.x_embedder = nn.Conv2d(total_in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, hidden_size) * 0.02, requires_grad=True)

        # EDM noise for target
        self.sigma_map = nn.Sequential(
            FourierEmbedding(256), 
            nn.Linear(256, hidden_size), 
            nn.SiLU(), 
            nn.Linear(hidden_size, hidden_size)
        )

        self.cond_noise_map = nn.Sequential(
            FourierEmbedding(256), 
            nn.Linear(256, hidden_size), 
            nn.SiLU(), 
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.act_embedder = nn.Embedding(num_actions, hidden_size)
        self.ctx_act_embedder = nn.Embedding(num_actions, hidden_size)
        self.ctx_act_proj = nn.Linear(hidden_size * context_frames, hidden_size)

        self.ctx_reward_embedder = nn.Linear(1, hidden_size)
        self.ctx_reward_proj = nn.Linear(hidden_size * context_frames, hidden_size)

        self.ctx_done_embedder = nn.Embedding(2, hidden_size) 
        self.ctx_done_proj = nn.Linear(hidden_size * context_frames, hidden_size)

        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)

        self.reward_head = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, 1)
        )
        self.done_head = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, x_noisy, sigma, context, cond_noise_level, target_action, context_actions, context_rewards, context_dones):
        # EDM preconditioning
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_noise = sigma.log() / 4.0
        c_in = c_in.view(-1, 1, 1, 1)
        c_skip = c_skip.view(-1, 1, 1, 1)
        c_out = c_out.view(-1, 1, 1, 1)
        F_x = c_in * x_noisy

        model_input = torch.cat([F_x, context], dim=1)

        # EDM Noise mapping
        sigma_vec = self.sigma_map(c_noise)

        cond_noise_vec = self.cond_noise_map(cond_noise_level.log() / 4.0)

        act_vec = self.act_embedder(target_action)
        ctx_act_vec = self.ctx_act_proj(self.ctx_act_embedder(context_actions).flatten(1))
        ctx_rew_vec = self.ctx_reward_proj(self.ctx_reward_embedder(context_rewards).flatten(1))
        ctx_done_vec = self.ctx_done_proj(self.ctx_done_embedder(context_dones).flatten(1))
        c = sigma_vec + cond_noise_vec + act_vec + ctx_act_vec + ctx_rew_vec + ctx_done_vec
        
        x = self.x_embedder(model_input).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x, c)

        scene_vector = x.mean(dim=1)
        pred_reward = self.reward_head(scene_vector)
        pred_done = self.done_head(scene_vector)

        x = self.final_layer(x, c)        
        F_out = unpatchify(x, self.in_channels)
        D_x = c_skip * x_noisy + c_out * F_out

        return D_x, pred_reward, pred_done

def train_mod_dit(dataset_path, num_actions, epochs=10, batch_size=32, val_split=0.1, 
                 in_channels=4, context_frames=4, hidden_size=384, depth=6, num_heads=6, 
                 device='cuda', input_size=8, patch_size=2):    
    
    full_dataset = AtariH5Dataset(dataset_path, context_len=context_frames)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    P_mean = -1.2
    P_std = 1.2
    sigma_data = 0.5

    model = ModDiT(
        input_size=input_size,
        patch_size=patch_size,
        in_channels=in_channels, 
        context_frames=context_frames, 
        hidden_size=hidden_size, 
        depth=depth, 
        num_heads=num_heads, 
        num_actions=num_actions,
        sigma_data=sigma_data
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0
        update_interval = max(1, len(train_loader) // 10)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", miniters=update_interval)
        
        for target, context, tgt_act, ctx_acts, reward, termination, ctx_rews, ctx_dones in pbar:
            target, context = target.to(device), context.to(device)
            tgt_act, ctx_acts = tgt_act.to(device), ctx_acts.to(device)
            tgt_rew = reward.to(device).view(-1, 1)
            tgt_done = termination.to(device).view(-1, 1)
            ctx_rews = ctx_rews.to(device)
            ctx_dones = ctx_dones.to(device)
            batch_size = target.shape[0]

            aug_sigma = torch.exp(torch.randn(batch_size, device=device) * P_std + P_mean).view(-1, 1, 1, 1)
            context_noisy = context + aug_sigma * torch.randn_like(context)
            cond_noise_level = aug_sigma.view(-1)

            rnd_normal = torch.randn([batch_size, 1, 1, 1], device=device)
            sigma = (rnd_normal * P_std + P_mean).exp()
            noise = torch.randn_like(target)
            target_noisy = target + sigma * noise

            D_x, pred_r, pred_d = model(target_noisy, sigma.view(-1), context_noisy, cond_noise_level, tgt_act, ctx_acts, ctx_rews, ctx_dones)
            weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2

            loss_img = (weight * ((D_x - target) ** 2)).mean()
            loss_r = mse_loss(pred_r, tgt_rew)
            loss_d = bce_loss(pred_d, tgt_done)
            loss = loss_img + loss_r + loss_d

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            if pbar.n % update_interval == 0:
                pbar.set_postfix(img=f"{loss_img.item():.2f}", r=f"{loss_r.item():.2f}", d=f"{loss_d.item():.2f}")
            
        avg_train_loss = train_loss_sum / len(train_loader)

        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for target, context, tgt_act, ctx_acts, reward, termination, ctx_rews, ctx_dones in val_loader:
                target, context = target.to(device), context.to(device)
                tgt_act, ctx_acts = tgt_act.to(device), ctx_acts.to(device)
                tgt_rew = reward.to(device).view(-1, 1)
                tgt_done = termination.to(device).view(-1, 1)
                ctx_rews = ctx_rews.to(device)
                ctx_dones = ctx_dones.to(device)
                batch_size = target.shape[0]


                aug_sigma = torch.exp(torch.randn(batch_size, device=device) * P_std + P_mean).view(-1, 1, 1, 1)
                context_noisy = context + aug_sigma * torch.randn_like(context)
                cond_noise_level = aug_sigma.view(-1)

                rnd_normal = torch.randn([batch_size, 1, 1, 1], device=device)
                sigma = (rnd_normal * P_std + P_mean).exp()
                noise = torch.randn_like(target)
                target_noisy = target + sigma * noise

                D_x, pred_r, pred_d = model(target_noisy, sigma.view(-1), context_noisy, cond_noise_level, tgt_act, ctx_acts, ctx_rews, ctx_dones)


                weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
                loss_img = (weight * ((D_x - target) ** 2)).mean()
                loss_r = mse_loss(pred_r, tgt_rew)
                loss_d = bce_loss(pred_d, tgt_done)
                
                total_loss = loss_img + loss_r + loss_d
                val_loss_sum += total_loss.item()
                

        avg_val_loss = val_loss_sum / len(val_loader) if len(val_loader) > 0 else 0
        save_msg = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "mod_dit.pt")
            save_msg = "--> Saved Best Model!"

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} {save_msg}")

    print("Training Complete.")


