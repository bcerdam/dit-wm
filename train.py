import argparse
import os
from diffusers.models import AutoencoderKL
# from atari_100k_dataset import env_rollout
from rollout_gen_p import env_rollout
from mod_dit import train_dit_wm

### TO DO ###

# Documentation
# parser DiT arquitecture args

### TO DO ###


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
    parser.add_argument('--val_split', type=float, default=0.1, help='Ratio of data used for validation (e.g., 0.1 for 10%)')
    parser.add_argument('--dataset_path', type=str, default='atari_dataset.h5', help='Path to HDF5 dataset')
    parser.add_argument('--delete_dataset', type=bool, default=False, help='If set, deletes existing dataset and starts fresh')
    
    args = parser.parse_args()

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

    
    print(f"Starting DiT Training | Batch Size: {args.batch_size} | Val Split: {args.val_split}")
    train_dit_wm(
        args.dataset_path, 
        epochs=args.n_epochs, 
        batch_size=args.batch_size, 
        val_split=args.val_split
    )
