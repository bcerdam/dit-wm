import gymnasium as gym
import ale_py
import h5py
import numpy as np
import torch
import multiprocessing as mp
from PIL import Image


def collect_rollout(env_name):
    import gymnasium as gym
    import ale_py

    gym.register_envs(ale_py)
    env = gym.make(env_name)
    obs, _ = env.reset()
    episode_over = False
    
    frames, actions, rewards, terminateds = [], [], [], []
    
    while not episode_over:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        img = Image.fromarray(obs).resize((64, 64))
        frames.append(np.array(img))
        
        actions.append(action)
        rewards.append(reward)
        terminateds.append(terminated)
        
        obs = next_obs
        episode_over = terminated or truncated
        
    env.close()
    return frames, actions, rewards, terminateds

def batch_encode(frames, vae, device):
    batch_np = np.stack(frames)
    
    with torch.no_grad():
        t = torch.tensor(batch_np).float().permute(0, 3, 1, 2).to(device)
        t = (t / 127.5) - 1.0
        chunk_size = 256
        latents_list = []
        for i in range(0, len(t), chunk_size):
            batch = t[i:i+chunk_size]
            dist = vae.encode(batch).latent_dist
            latents = dist.sample() * 0.18215
            latents_list.append(latents.cpu().numpy())
            
    return np.concatenate(latents_list, axis=0)

def env_rollout(env_name, n_rollouts, vae, dataset_path):
    with h5py.File(dataset_path, 'a') as f:
        if 'latents' not in f:
            f.create_dataset('latents', shape=(0, 4, 8, 8), maxshape=(None, 4, 8, 8), dtype='float32')
            f.create_dataset('actions', shape=(0,), maxshape=(None,), dtype='int32')
            f.create_dataset('rewards', shape=(0,), maxshape=(None,), dtype='float32')
            f.create_dataset('terminated', shape=(0,), maxshape=(None,), dtype='bool')

    print(f"Starting {n_rollouts} parallel rollouts...")

    ctx = mp.get_context('spawn')
    n_workers = min(20, n_rollouts) 
    with ctx.Pool(n_workers) as pool:
        results = pool.imap_unordered(collect_rollout, [env_name] * n_rollouts)
        for i, (frames, actions, rewards, term) in enumerate(results):
            
            latents = batch_encode(frames, vae, vae.device)
            n_steps = len(latents)

            with h5py.File(dataset_path, 'a') as f:
                f['latents'].resize(f['latents'].shape[0] + n_steps, axis=0)
                f['actions'].resize(f['actions'].shape[0] + n_steps, axis=0)
                f['rewards'].resize(f['rewards'].shape[0] + n_steps, axis=0)
                f['terminated'].resize(f['terminated'].shape[0] + n_steps, axis=0)

                f['latents'][-n_steps:] = latents
                f['actions'][-n_steps:] = np.array(actions)
                f['rewards'][-n_steps:] = np.array(rewards)
                f['terminated'][-n_steps:] = np.array(term)

            print(f"Rollout {i+1}/{n_rollouts} saved. ({n_steps} steps)")