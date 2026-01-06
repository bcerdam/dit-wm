import gymnasium as gym
import torch
import numpy as np
import cv2
import os
from gymnasium import spaces
from utils import edm_sampler

class DreamEnv(gym.Env):
    def __init__(self, dit_model, vae, device, pixel_space=False, context_frames=4, num_actions=4, num_steps=5, debug=True):
        super().__init__()
        self.model = dit_model
        self.vae = vae
        self.device = device
        self.pixel_space = pixel_space
        self.context_len = context_frames
        self.num_steps = num_steps

        self.debug = debug
        self.video_buffer = []
        self.video_counter = 0
        self.save_every_n_frames = 500
        self.output_dir = "dream_videos"
        if self.debug:
            os.makedirs(self.output_dir, exist_ok=True)

        if pixel_space:
            self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
            self.c, self.h, self.w = 3, 64, 64
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4, 8, 8), dtype=np.float32)
            self.c, self.h, self.w = 4, 8, 8

        self.action_space = spaces.Discrete(num_actions)
        self.context_frames = None
        self.context_actions = None
        self.warmup_buffer = [] 

    def set_warmup_buffer(self, real_obs_list):
        self.warmup_buffer = real_obs_list

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        if len(self.warmup_buffer) > 0:
            idx = np.random.randint(0, len(self.warmup_buffer))
            start_obs = self.warmup_buffer[idx] 
        else:
            start_obs = torch.zeros(self.context_len, self.c, self.h, self.w).to(self.device)

        self.context_frames = start_obs.clone().detach().float().to(self.device)
        self.context_actions = torch.zeros(1, self.context_len).long().to(self.device)
        
        return self._get_obs(), {}

    def step(self, action):
        if self.current_step % 10 == 0:
             print(f"- [DreamEnv] Step {self.current_step}")
             print(action)
        self.current_step += 1

        ctx_frames = self.context_frames.reshape(1, -1, self.h, self.w)
        ctx_actions = self.context_actions
        tgt_action = torch.tensor([action], device=self.device).long()
        
        next_frame, pred_r, pred_d = edm_sampler(
            self.model, ctx_frames, tgt_action, ctx_actions, 
            self.device, self.h, self.c, num_steps=self.num_steps
        )

        if self.debug:
            self._add_frame_to_video(next_frame)

        reward = pred_r.item()
        terminated = torch.sigmoid(pred_d).item() > 0.5
        truncated = False
        
        self.context_frames = torch.cat([self.context_frames[1:], next_frame], dim=0)
        self.context_actions = torch.cat([self.context_actions[:, 1:], tgt_action.unsqueeze(0)], dim=1)
        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    # Does not include logic for latent space
    def _get_obs(self):
        last_frame = self.context_frames[-1].detach().cpu().numpy()
        
        if self.pixel_space:
            last_frame = (last_frame + 1.0) / 2.0 * 255.0
            last_frame = np.clip(last_frame, 0, 255).astype(np.uint8)
            last_frame = np.transpose(last_frame, (1, 2, 0)) 
        
        return last_frame
    

    def _add_frame_to_video(self, frame_tensor):
        """Decodes latent/pixel tensor to a BGR numpy image and stores it."""
        with torch.no_grad():
            if self.pixel_space:
                # Pixel Space: Just denormalize [-1, 1] -> [0, 255]
                img = (frame_tensor[0].detach().cpu().numpy() + 1.0) / 2.0
                img = np.clip(img, 0, 1)
                img = (img * 255).astype(np.uint8)
                img = np.transpose(img, (1, 2, 0)) # H, W, C
            else:
                # Latent Space: Must Decode via VAE
                # Note: This slows down training slightly, but guarantees valid video
                latents = frame_tensor / 0.18215
                decoded = self.vae.decode(latents).sample
                img = (decoded[0].detach().cpu().numpy() / 2 + 0.5)
                img = np.clip(img, 0, 1)
                img = (img * 255).astype(np.uint8)
                img = np.transpose(img, (1, 2, 0)) # H, W, C

            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.video_buffer.append(img_bgr)

            # Flush to disk if buffer is full
            if len(self.video_buffer) >= self.save_every_n_frames:
                self._save_buffer_to_mp4()

    def _save_buffer_to_mp4(self):
        filename = os.path.join(self.output_dir, f"dream_debug_{self.video_counter}.mp4")
        height, width, _ = self.video_buffer[0].shape
        
        # Initialize Video Writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 15.0, (width, height))
        
        for frame in self.video_buffer:
            out.write(frame)
        
        out.release()
        print(f"[DreamEnv] Saved video chunk: {filename}")
        
        # Reset buffer
        self.video_buffer = []
        self.video_counter += 1