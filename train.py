import argparse
import os
import torch
import numpy as np
from diffusers.models import AutoencoderKL
from stable_baselines3 import PPO
from rollout_gen import env_rollout
from mod_dit import train_dit_wm, DiT_WM
from dream_env import DreamEnv
from atari_dataset import AtariH5Dataset
from utils import get_num_actions
from stable_baselines3.common.monitor import Monitor


DIT_CONFIGS = {
    'DiT-S': {'hidden_size': 384, 'depth': 12, 'num_heads': 6},
    'DiT-B': {'hidden_size': 768, 'depth': 12, 'num_heads': 12},
    'DiT-L': {'hidden_size': 1024, 'depth': 24, 'num_heads': 16},
    'DiT-XL': {'hidden_size': 1152, 'depth': 28, 'num_heads': 16},
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DiT-WM")
    
    parser.add_argument('--env_name', type=str, default='ALE/Breakout-v5', help='Atari environment ID')
    parser.add_argument('--n_rollouts', type=int, default=100, help='Rollouts to collect per policy step')

    parser.add_argument('--val_split', type=float, default=0.2, help='Ratio of data used for validation (e.g., 0.1 for 10%)')
    parser.add_argument('--n_epochs', type=int, default=10, help='Training epochs for DiT')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for DiT training')
    parser.add_argument('--model', type=str, default='DiT-S', choices=list(DIT_CONFIGS.keys()), help='Standard DiT config')
    parser.add_argument('--in_channels', type=int, default=3, help='Number of channels in latent space')
    parser.add_argument('--context_frames', type=int, default=4, help='Number of history frames')
    parser.add_argument('--patch_size', type=int, default=8, help='Size of image patches (use 2 for Latent, 4 or 8 for Pixel)')
    parser.add_argument('--hidden_size', type=int, default=384, help='Transformer embedding dimension')
    parser.add_argument('--depth', type=int, default=6, help='Number of DiT blocks')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of attention heads')

    parser.add_argument('--denoising_steps', type=int, default=3, help='Number of denoising steps')


    parser.add_argument('--dataset_path', type=str, default='atari_dataset.h5', help='Path to HDF5 dataset')


    parser.add_argument('--delete_policy', action='store_true', default=True, help='If set, deletes existing PPO agent')
    parser.add_argument('--keep_policy', action='store_false', dest='delete_policy', help='Keep existing policy')
    
    parser.add_argument('--delete_dataset', action='store_true', default=True, help='If set, deletes existing dataset')
    parser.add_argument('--keep_dataset', action='store_false', dest='delete_dataset', help='Keep existing dataset')

    parser.add_argument('--delete_wm', action='store_true', default=True, help='If set, deletes existing DiT weights (dit_wm.pt)')
    parser.add_argument('--keep_wm', action='store_false', dest='delete_wm', help='Keep existing DiT weights')

    parser.add_argument('--clear_dataset_per_loop', action='store_true', default=True,
                        help='If set, deletes the dataset at the start of every MBRL loop to train only on fresh data')
    
    parser.add_argument('--pixel_space', type=bool, default=True, help='If set, trains on 64x64 RGB pixels instead of VAE latents')
    
    parser.add_argument('--mbrl_loops', type=int, default=100, help='Number of Collect-Train-Dream loops')
    parser.add_argument('--ppo_steps', type=int, default=1000, help='Steps to train PPO agent in dream')
    
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ENV_NAME = args.env_name
    NUM_ACTIONS = get_num_actions(ENV_NAME)
    N_ROLLOUTS = args.n_rollouts
    N_EPOCHS = args.n_epochs
    DATASET_PATH = args.dataset_path
    POLICY_PATH = "ppo_agent.zip"
    WM_PATH = "dit_wm.pt"

    if args.model:
        config = DIT_CONFIGS[args.model]
        print(f"- Using configuration for {args.model}")
        args.hidden_size = config['hidden_size']
        args.depth = config['depth']
        args.num_heads = config['num_heads']

    if args.delete_dataset and os.path.exists(args.dataset_path):
        os.remove(args.dataset_path)
        print(f"- Deleted existing dataset: {args.dataset_path}")
    
    if args.delete_policy and os.path.exists(POLICY_PATH):
        os.remove(POLICY_PATH)
        print(f"- Deleted existing policy: {POLICY_PATH}")

    if args.delete_wm and os.path.exists(WM_PATH):
        os.remove(WM_PATH)
        print(f"- Deleted existing World Model: {WM_PATH}")

    if args.pixel_space:
        VAE = None 
        args.in_channels = 3 
        input_size = 64
    else:
        VAE = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to('cuda')
        input_size = 8

    if not os.path.exists(POLICY_PATH):
        dummy_wm = DiT_WM(
            num_actions=NUM_ACTIONS,
            input_size=input_size, 
            patch_size=args.patch_size, 
            in_channels=args.in_channels, 
            context_frames=args.context_frames,
            hidden_size=args.hidden_size, 
            depth=args.depth, 
            num_heads=args.num_heads
        ).to(device)

        dummy_env = DreamEnv(dummy_wm, 
                             VAE, 
                             device, 
                             args.pixel_space, 
                             context_frames=args.context_frames, 
                             num_steps=args.denoising_steps,
                             num_actions=NUM_ACTIONS)
        
        agent = PPO("CnnPolicy", dummy_env, device='cpu')
        agent.save(POLICY_PATH)
        del dummy_wm, dummy_env, agent


    for loop in range(args.mbrl_loops):
        print(f"\n" + "="*40)
        print(f"      STARTING MBRL LOOP {loop+1}/{args.mbrl_loops}")
        print("="*40)

        if args.clear_dataset_per_loop and os.path.exists(DATASET_PATH):
            os.remove(DATASET_PATH)
            print(f"- [MBRL Loop {loop+1}] Cleared dataset (Fresh Rollouts Only).")

        print(f"- Collecting {N_ROLLOUTS} rollouts with policy: {POLICY_PATH}")
        env_rollout(ENV_NAME, N_ROLLOUTS, VAE, DATASET_PATH, policy_path=POLICY_PATH)

        print("- Training DiT World Model...")
        train_dit_wm(
            args.dataset_path, 
            num_actions=NUM_ACTIONS,
            epochs=N_EPOCHS, 
            batch_size=args.batch_size, 
            val_split=args.val_split,
            in_channels=args.in_channels,
            context_frames=args.context_frames,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            input_size=input_size, 
            patch_size=args.patch_size
        )

        # De aqui en adelante.
        print("- PPO currently dreaming...")
        
        wm_model = DiT_WM(
            num_actions=NUM_ACTIONS,
            input_size=input_size, 
            patch_size=args.patch_size, 
            in_channels=args.in_channels, 
            context_frames=args.context_frames,
            hidden_size=args.hidden_size, 
            depth=args.depth, 
            num_heads=args.num_heads
        ).to(device)
        wm_model.load_state_dict(torch.load("dit_wm.pt"))
        wm_model.eval()

        dream_env = DreamEnv(wm_model, 
                             VAE, 
                             device, 
                             args.pixel_space, 
                             context_frames=args.context_frames, 
                             num_steps=args.denoising_steps, 
                             num_actions=NUM_ACTIONS)
        dream_env = Monitor(dream_env)
        
        print("- Warmup buffer...")
        ds = AtariH5Dataset(DATASET_PATH, context_len=args.context_frames)
        warmup_obs = []
        indices = np.linspace(0, len(ds)-1, min(len(ds), 500), dtype=int)

        for idx in indices:
            ctx_data = ds[idx][1] 
            ctx_data = ctx_data.view(args.context_frames, args.in_channels, input_size, input_size)
            warmup_obs.append(ctx_data.to(device))
        
        dream_env.unwrapped.set_warmup_buffer(warmup_obs)

        if hasattr(ds, 'close'):
            ds.close()
        if hasattr(ds, 'file'):
            ds.file.close()
        del ds

        print(f"- Optimizing agent for {args.ppo_steps} steps...")
        agent = PPO.load(POLICY_PATH, env=dream_env, verbose=1, device='cpu')
        agent.learn(total_timesteps=args.ppo_steps)
        agent.save(POLICY_PATH)
        print("- Agent updated and saved.")

