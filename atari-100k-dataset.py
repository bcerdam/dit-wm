import gymnasium as gym
import ale_py
from diffusers.models import AutoencoderKL
from plot_utils import plot_atari_frame
from vae import encode_observation, decode_latent


'''
env_rollout(): Collects dataset of enviroment with current policy

env_name: Atari enviroment name
n_rollouts: Number of enviroment rollouts
vae: Off the shelf VAE
'''
def env_rollout(env_name, n_rollouts, vae):

    gym.register_envs(ale_py)
    env = gym.make(ENV_NAME)
    observation, info = env.reset()

    episode_over = False
    while not episode_over:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        # Encode
        latent = encode_observation(observation, vae)

        # Decode
        # decoded_latent = decode_latent(latent, vae)
        # plot_atari_frame(decoded_latent)

        break

        episode_over = terminated or truncated
    env.close()


ENV_NAME = 'ALE/Breakout-v5'
VAE = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to('cuda')
N_ROLLOUTS = 1

env_rollout(ENV_NAME, N_ROLLOUTS, VAE)
