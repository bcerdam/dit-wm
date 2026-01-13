import gymnasium as gym
import torch
import numpy as np
import h5py
from gymnasium import spaces
from mod_dit import ModDiT
from utils import edm_sampler
from diffusers.models import AutoencoderKL

class DreamEnv(gym.Env):
    def __init__(self, model_path, dataset_path, device='cuda', context_frames=64, 
                 num_actions=4, denoising_steps=5, pixel_space=False,
                 input_size=8, in_channels=4, patch_size=2,
                 hidden_size=384, depth=6, num_heads=6, max_steps=1000):
        super().__init__()
        self.render_mode = 'rgb_array'
        
        self.max_steps = max_steps
        self.current_step = 0
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 10}
        self.last_obs = None

        self.device = device
        self.context_frames = context_frames
        self.denoising_steps = denoising_steps
        self.pixel_space = pixel_space
        
        self.input_size = input_size
        self.in_channels = in_channels

        self.dataset_file = h5py.File(dataset_path, 'r')
        self.observations = self.dataset_file['observations']
        self.actions = self.dataset_file['actions']
        self.rewards = self.dataset_file['rewards']
        self.terminated = self.dataset_file['terminated']
        self.dataset_len = len(self.observations)
        
        if not self.pixel_space:
            print(f"Loading VAE for decoding latents...")
            self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
            self.vae.eval()
        else:
            self.vae = None

        self.model = ModDiT(
            num_actions=num_actions,
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            context_frames=context_frames,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads
        ).to(device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        self.action_space = spaces.Discrete(num_actions)
        
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(64, 64, 3), 
            dtype=np.uint8
        )

        self.current_latents = None
        self.current_actions = None
        self.current_rewards = None
        self.current_dones = None

        sigma_min, sigma_max, rho = 0.002, 80.0, 7.0
        step_indices = torch.arange(denoising_steps, dtype=torch.float64, device=device)
        self.t_steps = (sigma_max ** (1 / rho) + step_indices / (denoising_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        self.t_steps = torch.cat([self.t_steps, torch.zeros_like(self.t_steps[:1])]).float()

    def _get_context(self, idx):
        start = idx
        end = idx + self.context_frames
        
        raw_obs = self.observations[start:end]
        
        if self.pixel_space and raw_obs.dtype == 'uint8':
            obs_tensor = torch.tensor(raw_obs).float().to(self.device)
            obs_tensor = (obs_tensor / 127.5) - 1.0
        else:
            obs_tensor = torch.tensor(raw_obs).float().to(self.device)

        acts = torch.tensor(self.actions[start:end]).long().to(self.device)
        rews = torch.tensor(self.rewards[start:end]).float().view(-1, 1).to(self.device)
        dones = torch.tensor(self.terminated[start:end]).float().view(-1, 1).to(self.device)
        
        return obs_tensor, acts, rews, dones
    
    def _decode_to_obs(self, latent_tensor):
        with torch.no_grad():
            if self.pixel_space:
                img = (latent_tensor + 1.0) / 2.0 * 255.0
            else:
                lat = latent_tensor.unsqueeze(0) / 0.18215
                img = self.vae.decode(lat).sample[0] # [3, 64, 64]
                img = (img / 2 + 0.5) * 255.0

            img = img.clamp(0, 255).byte()
            img_np = img.permute(1, 2, 0).cpu().numpy() # [64, 64, 3]
            return img_np

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        valid = False
        while not valid:
            idx = np.random.randint(0, self.dataset_len - self.context_frames - 1)
            if np.sum(self.terminated[idx : idx + self.context_frames]) == 0:
                valid = True

        self.current_latents, self.current_actions, self.current_rewards, self.current_dones = self._get_context(idx)
        
        obs = self._decode_to_obs(self.current_latents[-1])
        self.last_obs = obs
        return obs, {}

    def step(self, action):
        self.current_step += 1

        ctx_latents = self.current_latents.reshape(1, -1, self.input_size, self.input_size)
        ctx_actions = self.current_actions.unsqueeze(0)
        ctx_rewards = self.current_rewards.unsqueeze(0)
        ctx_dones = self.current_dones.long().unsqueeze(0)
        
        tgt_action = torch.tensor([action], device=self.device).long()

        D_x, pred_r, pred_d = edm_sampler(
            self.model, 
            ctx_latents, 
            tgt_action, 
            ctx_actions, 
            ctx_rewards, 
            ctx_dones, 
            self.device, 
            self.input_size, 
            self.in_channels, 
            num_steps=self.denoising_steps
        )

        next_latent = D_x[0]
        reward = pred_r.item()
        prob_done = torch.sigmoid(pred_d).item()
        terminated = prob_done > 0.5 
        truncated = False 

        self.current_latents = torch.cat([self.current_latents[1:], D_x], dim=0)
        self.current_actions = torch.cat([self.current_actions[1:], tgt_action], dim=0)
        
        new_rew = torch.tensor([[reward]], device=self.device)
        new_done = torch.tensor([[1.0 if terminated else 0.0]], device=self.device)
        
        self.current_rewards = torch.cat([self.current_rewards[1:], new_rew], dim=0)
        self.current_dones = torch.cat([self.current_dones[1:], new_done], dim=0)

        obs_pixels = self._decode_to_obs(next_latent)
        self.last_obs = obs_pixels

        if self.current_step >= self.max_steps:
            truncated = True
        else:
            truncated = False

        return obs_pixels, reward, terminated, truncated, {}
    
    def render(self):
        return self.last_obs

    def close(self):
        if hasattr(self, 'dataset_file'):
            self.dataset_file.close()