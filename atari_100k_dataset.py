import gymnasium as gym
import ale_py
import h5py
import numpy as np
from utils import plot_atari_frame
from vae import encode_observation, decode_latent


'''
env_rollout(): Collects dataset of enviroment with current policy

env_name: Atari enviroment name
n_rollouts: Number of enviroment rollouts
vae: Off the shelf VAE
dataset_path: path of atari .h5 dataset
'''
def env_rollout(env_name, n_rollouts, vae, dataset_path):
    gym.register_envs(ale_py)
    env = gym.make(env_name)
    DATASET_PATH = dataset_path

    with h5py.File(DATASET_PATH, 'a') as f:
        if 'latents' not in f:
            f.create_dataset('latents', shape=(0, 4, 8, 8), maxshape=(None, 4, 8, 8), dtype='float32')
            f.create_dataset('actions', shape=(0,), maxshape=(None,), dtype='int32')
            f.create_dataset('rewards', shape=(0,), maxshape=(None,), dtype='float32')
            f.create_dataset('terminated', shape=(0,), maxshape=(None,), dtype='bool')

    for rollout in range(n_rollouts):
        observation, info = env.reset()
        episode_over = False

        b_latents, b_actions, b_rewards, b_term = [], [], [], []
        while not episode_over:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

            # Encode
            latent = encode_observation(observation, vae)

            b_latents.append(latent)
            b_actions.append(action)
            b_rewards.append(reward)
            b_term.append(terminated)

            # Decode
            # decoded_latent = decode_latent(latent, vae)
            # plot_atari_frame(decoded_latent)

            episode_over = terminated or truncated

        with h5py.File(DATASET_PATH, 'a') as f:
            new_latents = np.array(b_latents).squeeze(1)
            n_steps = len(new_latents)

            f['latents'].resize(f['latents'].shape[0] + n_steps, axis=0)
            f['actions'].resize(f['actions'].shape[0] + n_steps, axis=0)
            f['rewards'].resize(f['rewards'].shape[0] + n_steps, axis=0)
            f['terminated'].resize(f['terminated'].shape[0] + n_steps, axis=0)

            f['latents'][-n_steps:] = new_latents
            f['actions'][-n_steps:] = np.array(b_actions)
            f['rewards'][-n_steps:] = np.array(b_rewards)
            f['terminated'][-n_steps:] = np.array(b_term)
        print(f"Rollout {rollout+1}/{n_rollouts} saved.")

    env.close()
