import gymnasium as gym
import ale_py
import torch


DIT_CONFIGS = {
    'DiT-S': {'hidden_size': 384, 'depth': 12, 'num_heads': 6},
    'DiT-B': {'hidden_size': 768, 'depth': 12, 'num_heads': 12},
    'DiT-L': {'hidden_size': 1024, 'depth': 24, 'num_heads': 16},
    'DiT-XL': {'hidden_size': 1152, 'depth': 28, 'num_heads': 16},
}


def get_num_actions(env_name):
    gym.register_envs(ale_py)
    env = gym.make(env_name)
    n = env.action_space.n
    env.close()
    return n


def unpatchify(x, channels):
    patch_dim = x.shape[-1]
    p = int((patch_dim // channels) ** 0.5)
    h = w = int(x.shape[1] ** 0.5)
    x = x.reshape(shape=(x.shape[0], h, w, p, p, channels))
    x = torch.einsum('nhwpqc->nchpwq', x)
    return x.reshape(shape=(x.shape[0], channels, h * p, h * p))