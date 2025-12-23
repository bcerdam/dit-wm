import gymnasium as gym
import ale_py
from plot_utils import plot_atari_frame

ENV_NAME = 'ALE/Breakout-v5'

gym.register_envs(ale_py)
env = gym.make(ENV_NAME)
obs, info = env.reset()

episode_over = False
while not episode_over:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    # print(f'Observation shape: {obs.shape}')
    plot_atari_frame(obs)

    episode_over = terminated or truncated
    break
env.close()