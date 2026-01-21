import gymnasium as gym
import ale_py
import h5py
import numpy as np
import torch
import cv2
from stable_baselines3.common.atari_wrappers import FireResetEnv
from vae import train_vae, VAE


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
            z_mean, z_log_var = vae.encoder(batch)
            latents = vae.sampler(z_mean, z_log_var)
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


def env_rollout(env_name, n_steps, dataset_path, data_shape, dtype, resize_resolution):
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
    all_episodes_observations, all_episodes_actions, all_episodes_rewards, all_episodes_termination_status = [], [], [], []
    print(f'Gathering {n_steps} steps... (1/3)')
    while collected_steps < n_steps:
        observations, actions, rewards, termination_status = run_episode(env=env, resize_resolution=resize_resolution)

        all_episodes_observations.append(np.array(observations))
        all_episodes_actions.append(actions)
        all_episodes_rewards.append(rewards)
        all_episodes_termination_status.append(termination_status)
        
        step_len = len(observations)
        collected_steps += step_len
        rollout_idx += 1

    env.close()
    return all_episodes_observations, all_episodes_actions, all_episodes_rewards, all_episodes_termination_status

    
def process_observations(n_steps, all_episodes_observations, all_episodes_actions, all_episodes_rewards, all_episodes_termination_status, pixel_space,
                         latent_channel_dim, latent_spatial_dim, resize_resolution, vae_weights_path, dataset_path, device='cuda'):
    
    print(f'Processing and saving {n_steps} steps... (2/3)')
    for episode_idx in range(len(all_episodes_observations)):
        if pixel_space:
            data_batch = process_pixels_only(all_episodes_observations[episode_idx])
        else:
            vae = VAE(latent_channel_dim=latent_channel_dim, 
                      latent_spatial_dim=latent_spatial_dim, 
                      observation_resolution=resize_resolution).to(device)
    
            vae.load_state_dict(torch.load(vae_weights_path, map_location=device))
            vae.eval()
            data_batch = batch_encode(all_episodes_observations[episode_idx], vae, device)
        
        save_to_h5(dataset_path, 
                   data_batch, 
                   all_episodes_actions[episode_idx], 
                   all_episodes_rewards[episode_idx], 
                   all_episodes_termination_status[episode_idx])
    print(f'Finished data collection (3/3)')