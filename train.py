import argparse
import os
from diffusers.models import AutoencoderKL
from rollout_gen import env_rollout
from mod_dit import train_mod_dit
from utils import get_num_actions, DIT_CONFIGS


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DiT-WM")
    
    parser.add_argument('--env_name', type=str, default='ALE/Breakout-v5', help='Atari environment ID')
    parser.add_argument('--n_steps', type=int, default=100000, help='Total environment steps to collect')

    parser.add_argument('--val_split', type=float, default=0.2, help='Ratio of data used for validation (e.g., 0.1 for 10%)')
    parser.add_argument('--dit_n_epochs', type=int, default=10, help='Training epochs for Dynamics Model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for DiT training')
    parser.add_argument('--model', type=str, default='DiT-B', choices=list(DIT_CONFIGS.keys()), help='Standard DiT config')
    parser.add_argument('--context_frames', type=int, default=4, help='Number of history frames')
    parser.add_argument('--patch_size', type=int, default=8, help='Size of image patches (use 2 for Latent, 4 or 8 for Pixel)')
    parser.add_argument('--hidden_size', type=int, default=384, help='Transformer embedding dimension')
    parser.add_argument('--depth', type=int, default=6, help='Number of DiT blocks')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of attention heads')

    parser.add_argument('--dataset_path', type=str, default='atari_dataset.h5', help='Path to HDF5 dataset')
    parser.add_argument('--weights_path', type=str, default='mod_dit.pt', help='Path to model weights (denoise mode)')
    
    parser.add_argument('--delete_dataset', action='store_true', default=True, help='If set, deletes existing dataset')
    parser.add_argument('--keep_dataset', action='store_false', dest='delete_dataset', help='Keep existing dataset')

    parser.add_argument('--delete_dit_weights', action='store_true', default=True, help='If set, deletes existing DiT weights')
    parser.add_argument('--keep_dit_weights', action='store_false', dest='delete_dit_weights', help='Keep existing DiT weights')
    
    parser.add_argument('--pixel_space', type=bool, default=True, help='If set, trains on 64x64 RGB pixels instead of VAE latents')
    
    args = parser.parse_args()

    ENV_NAME = args.env_name
    NUM_ACTIONS = get_num_actions(ENV_NAME)
    N_STEPS = args.n_steps
    DIT_EPOCHS = args.dit_n_epochs
    DATASET_PATH = args.dataset_path
    DIT_WEIGHTS_PATH = args.weights_path

    if args.model:
        config = DIT_CONFIGS[args.model]
        print(f"- Using configuration for {args.model}")
        args.hidden_size = config['hidden_size']
        args.depth = config['depth']
        args.num_heads = config['num_heads']

    if args.delete_dataset and os.path.exists(args.dataset_path):
        os.remove(args.dataset_path)
        print(f"- Deleted existing dataset: {args.dataset_path}")

    if args.delete_dit_weights and os.path.exists(DIT_WEIGHTS_PATH):
        os.remove(DIT_WEIGHTS_PATH)
        print(f"- Deleted existing Dynamics Model weights: {DIT_WEIGHTS_PATH}")

    if args.pixel_space:
        VAE = None 
        in_channels = 3
        input_size = 64
    else:
        VAE = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to('cuda')
        in_channels = 4
        input_size = 8

    env_rollout(ENV_NAME, N_STEPS, VAE, DATASET_PATH, policy_path=None)
    train_mod_dit(
        args.dataset_path, 
        num_actions=NUM_ACTIONS,
        epochs=DIT_EPOCHS, 
        batch_size=args.batch_size, 
        val_split=args.val_split,
        in_channels=in_channels,
        context_frames=args.context_frames,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        input_size=input_size, 
        patch_size=args.patch_size
    )



