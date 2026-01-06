import gymnasium as gym
import ale_py
import h5py
import numpy as np
import torch
import cv2
import os
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import FireResetEnv, EpisodicLifeEnv


def run_episode(env, model, pixel_space, max_steps=1000):
    env = FireResetEnv(env)
    obs, _ = env.reset()
    
    frames, actions, rewards, terminateds = [], [], [], []
    step_count = 0
    episode_over = False

    while not episode_over:
        img_obs = cv2.resize(obs, (64, 64), interpolation=cv2.INTER_AREA)
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)

        frames.append(img_obs)
        actions.append(action)
        rewards.append(reward)
        terminateds.append(terminated)
        
        obs = next_obs
        step_count += 1
        
        if step_count >= max_steps:
            truncated = True
            
        episode_over = terminated or truncated
        
    return frames, actions, rewards, terminateds

def batch_encode(frames, vae, device):
    batch_np = np.stack(frames)
    
    with torch.no_grad():
        t = torch.tensor(batch_np).float().permute(0, 3, 1, 2).to(device)
        t = (t / 127.5) - 1.0
        
        latents_list = []
        for i in range(0, len(t), 256):
            batch = t[i:i+256]
            dist = vae.encode(batch).latent_dist
            latents = dist.sample() * 0.18215
            latents_list.append(latents.cpu().numpy())
            
    return np.concatenate(latents_list, axis=0)


def process_pixels_only(frames):
    batch_np = np.stack(frames)
    return np.transpose(batch_np, (0, 3, 1, 2))


def save_to_h5(dataset_path, obs_data, actions, rewards, terminated):
    n_new_steps = len(obs_data)

    with h5py.File(dataset_path, 'a') as f:
        current_len = f['observations'].shape[0]
        new_len = current_len + n_new_steps
        
        f['observations'].resize(new_len, axis=0)
        f['actions'].resize(new_len, axis=0)
        f['rewards'].resize(new_len, axis=0)
        f['terminated'].resize(new_len, axis=0)

        f['observations'][current_len:] = obs_data
        f['actions'][current_len:] = np.array(actions)
        f['rewards'][current_len:] = np.array(rewards)
        f['terminated'][current_len:] = np.array(terminated)


def env_rollout(env_name, n_steps, vae, dataset_path, policy_path=None):
    gym.register_envs(ale_py)
    
    pixel_space = (vae is None)
    if pixel_space:
        data_shape = (3, 64, 64)
        dtype = 'uint8'
    else:
        data_shape = (4, 8, 8)
        dtype = 'float32'

    with h5py.File(dataset_path, 'a') as f:
        if 'observations' not in f:
            f.create_dataset('observations', shape=(0, *data_shape), maxshape=(None, *data_shape), dtype=dtype)
            f.create_dataset('actions', shape=(0,), maxshape=(None,), dtype='int32')
            f.create_dataset('rewards', shape=(0,), maxshape=(None,), dtype='float32')
            f.create_dataset('terminated', shape=(0,), maxshape=(None,), dtype='bool')

    env = gym.make(env_name, render_mode='rgb_array')
    model = None
    collected_steps = 0
    rollout_idx = 0
    print(f"Starting data collection. Target: {n_steps} steps.")

    while collected_steps < n_steps:
        frames, actions, rewards, term = run_episode(env, model, pixel_space)
        
        if pixel_space:
            data_batch = process_pixels_only(frames)
        else:
            data_batch = batch_encode(frames, vae, vae.device)
        
        save_to_h5(dataset_path, data_batch, actions, rewards, term)
        step_len = len(frames)
        collected_steps += step_len
        rollout_idx += 1
        print(f"Rollout {rollout_idx}: Collected {step_len} steps. Total: {collected_steps}/{n_steps}")
            
    env.close()