import gymnasium as gym
import ale_py
import h5py
import numpy as np
import torch
import cv2
from stable_baselines3.common.atari_wrappers import FireResetEnv


def run_episode(env, resize_resolution):
    env = FireResetEnv(env)
    obs, info = env.reset()
    
    observations, actions, rewards, termination_status = [], [], [], []
    episode_over = False
    while not episode_over:
        img_obs = cv2.resize(obs, (resize_resolution, resize_resolution), interpolation=cv2.INTER_AREA)

        # Random policy
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        observations.append(img_obs)
        actions.append(action)
        rewards.append(reward)
        termination_status.append(terminated)
        
        obs = next_obs
        episode_over = terminated or truncated
        
    return observations, actions, rewards, termination_status

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


def process_pixels_only(observations):
    batch_np = np.stack(observations)
    batch_np_T = np.transpose(batch_np, (0, 3, 1, 2))
    return batch_np_T


def save_to_h5(dataset_path, obs_data, actions, rewards, termination_status):
    n_new_steps = len(obs_data)

    with h5py.File(dataset_path, 'a') as f:
        current_len = f['observations'].shape[0]
        new_len = current_len + n_new_steps
        
        f['observations'].resize(new_len, axis=0)
        f['actions'].resize(new_len, axis=0)
        f['rewards'].resize(new_len, axis=0)
        f['termination_status'].resize(new_len, axis=0)

        f['observations'][current_len:] = obs_data
        f['actions'][current_len:] = np.array(actions)
        f['rewards'][current_len:] = np.array(rewards)
        f['termination_status'][current_len:] = np.array(termination_status)


def env_rollout(env_name, n_steps, vae, dataset_path, pixel_space, data_shape, dtype, resize_resolution):
    gym.register_envs(ale_py)

    with h5py.File(dataset_path, 'a') as f:
        if 'observations' not in f:
            f.create_dataset('observations', shape=(0, *data_shape), maxshape=(None, *data_shape), dtype=dtype)
            f.create_dataset('actions', shape=(0,), maxshape=(None,), dtype='int64')
            f.create_dataset('rewards', shape=(0,), maxshape=(None,), dtype='float')
            f.create_dataset('termination_status', shape=(0,), maxshape=(None,), dtype='bool')

    env = gym.make(env_name, render_mode='rgb_array')

    collected_steps = 0
    rollout_idx = 0
    while collected_steps < n_steps:
        observations, actions, rewards, termination_status = run_episode(env=env, resize_resolution=resize_resolution)
        
        if pixel_space:
            data_batch = process_pixels_only(observations)
        else:
            data_batch = batch_encode(observations, vae, vae.device)
        
        save_to_h5(dataset_path, data_batch, actions, rewards, termination_status)
        step_len = len(observations)
        collected_steps += step_len
        rollout_idx += 1
        print(f"Rollout {rollout_idx}: Collected {step_len} steps. Total: {collected_steps}/{n_steps}")
            
    env.close()