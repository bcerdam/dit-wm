import argparse
import os
from diffusers.models import AutoencoderKL
from rollout_gen_p import env_rollout
from mod_dit import train_dit_wm

DIT_CONFIGS = {
    'DiT-S': {'hidden_size': 384, 'depth': 12, 'num_heads': 6},
    'DiT-B': {'hidden_size': 768, 'depth': 12, 'num_heads': 12},
    'DiT-L': {'hidden_size': 1024, 'depth': 24, 'num_heads': 16},
    'DiT-XL': {'hidden_size': 1152, 'depth': 28, 'num_heads': 16},
}

'''
    1. data collection (100 rollouts per policy)
    2. DiT train on whole dataset (Includes rollouts with init policy and improved policy)
    3. Reward and Termination heads
    4. PPO using DiT as dynamics model
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DiT-WM")
    
    parser.add_argument('--env_name', type=str, default='ALE/Breakout-v5', help='Atari environment ID')
    parser.add_argument('--n_rollouts', type=int, default=100, help='Rollouts to collect per policy step')
    parser.add_argument('--n_epochs', type=int, default=50, help='Training epochs for DiT')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for DiT training')
    parser.add_argument('--model', type=str, default='DiT-S', choices=list(DIT_CONFIGS.keys()), help='Standard DiT config')
    parser.add_argument('--in_channels', type=int, default=4, help='Number of channels in latent space')
    parser.add_argument('--context_frames', type=int, default=4, help='Number of history frames')
    parser.add_argument('--hidden_size', type=int, default=384, help='Transformer embedding dimension')
    parser.add_argument('--depth', type=int, default=6, help='Number of DiT blocks')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--val_split', type=float, default=0.1, help='Ratio of data used for validation (e.g., 0.1 for 10%)')
    parser.add_argument('--dataset_path', type=str, default='atari_dataset.h5', help='Path to HDF5 dataset')
    parser.add_argument('--delete_dataset', type=bool, default=False, help='If set, deletes existing dataset and starts fresh')
    
    args = parser.parse_args()

    if args.model:
        config = DIT_CONFIGS[args.model]
        print(f"(!) Using standard configuration for {args.model}")
        args.hidden_size = config['hidden_size']
        args.depth = config['depth']
        args.num_heads = config['num_heads']

    if args.delete_dataset:
        if os.path.exists(args.dataset_path):
            os.remove(args.dataset_path)
            print(f"(!) Deleted existing dataset: {args.dataset_path}")
        else:
            print(f"(!) --delete_dataset set, but {args.dataset_path} not found.")

    ENV_NAME = args.env_name
    N_ROLLOUTS = args.n_rollouts
    N_EPOCHS = args.n_epochs
    DATASET_PATH = args.dataset_path

    VAE = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to('cuda')
    env_rollout(ENV_NAME, N_ROLLOUTS, VAE, DATASET_PATH)

    train_dit_wm(
        args.dataset_path, 
        epochs=args.n_epochs, 
        batch_size=args.batch_size, 
        val_split=args.val_split,
        in_channels=args.in_channels,
        context_frames=args.context_frames,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads
    )