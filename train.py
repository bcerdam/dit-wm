import argparse
import os
from diffusers.models import AutoencoderKL
from rollout_gen import env_rollout
from mod_dit import train_mod_dit
from utils import get_num_actions, DIT_CONFIGS
from agent_train import train_agent_in_dream


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DiT-WM")
    
    parser.add_argument('--env_name', type=str, default='ALE/Breakout-v5', help='Atari environment ID')
    parser.add_argument('--update_wm_epochs', type=int, default=50, help='Number of outer loops (Collect -> Train WM -> Train Agent)')
    parser.add_argument('--agent_steps', type=int, default=2048*2, help='Steps to train PPO per loop')
    parser.add_argument('--denoising_steps', type=int, default=5, help='Number of EDM steps for DreamEnv (Inference)')
    parser.add_argument('--n_steps', type=int, default=2000, help='Total environment steps to collect')

    parser.add_argument('--val_split', type=float, default=0.2, help='Ratio of data used for validation (e.g., 0.1 for 10%)')
    parser.add_argument('--dit_n_epochs', type=int, default=5, help='Training epochs for Dynamics Model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for DiT training')
    parser.add_argument('--model', type=str, default='DiT-S', choices=list(DIT_CONFIGS.keys()), help='Standard DiT config')
    parser.add_argument('--context_frames', type=int, default=4, help='Number of history frames')
    parser.add_argument('--patch_size', type=int, default=2, help='Size of image patches (use 2 for Latent, 4 or 8 for Pixel)')
    parser.add_argument('--hidden_size', type=int, default=384, help='Transformer embedding dimension')
    parser.add_argument('--depth', type=int, default=6, help='Number of DiT blocks')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of attention heads')

    parser.add_argument('--dataset_path', type=str, default='atari_dataset.h5', help='Path to HDF5 dataset')
    parser.add_argument('--weights_path', type=str, default='mod_dit.pt', help='Path to model weights (denoise mode)')
    parser.add_argument('--agent_path', type=str, default='ppo_dream_agent.zip', help='Path to save PPO agent')
    
    parser.add_argument('--delete_dataset', action='store_true', default=True, help='If set, deletes existing dataset')
    parser.add_argument('--keep_dataset', action='store_false', dest='delete_dataset', help='Keep existing dataset')

    parser.add_argument('--delete_dit_weights', action='store_true', default=True, help='If set, deletes existing DiT weights')
    parser.add_argument('--keep_dit_weights', action='store_false', dest='delete_dit_weights', help='Keep existing DiT weights')

    parser.add_argument('--delete_agent', action='store_true', default=True, help='If set, deletes existing PPO agent at start')
    parser.add_argument('--keep_agent', action='store_false', dest='delete_agent', help='Keep existing PPO agent')
    
    parser.add_argument('--pixel_space', action='store_true', help='If set, trains on 64x64 RGB pixels instead of VAE latents')

    
    args = parser.parse_args()

    ENV_NAME = args.env_name
    NUM_ACTIONS = get_num_actions(ENV_NAME)
    N_STEPS = args.n_steps
    DIT_EPOCHS = args.dit_n_epochs
    DATASET_PATH = args.dataset_path
    DIT_WEIGHTS_PATH = args.weights_path
    AGENT_PATH = args.agent_path

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

    if args.delete_agent and os.path.exists(AGENT_PATH):
        os.remove(AGENT_PATH)
        print(f"- Deleted existing Agent: {AGENT_PATH}")

    if args.pixel_space:
        VAE = None 
        in_channels = 3
        input_size = 64
    else:
        VAE = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to('cuda')
        in_channels = 4
        input_size = 8

    for iteration in range(args.update_wm_epochs):
        print(f"\n" + "-"*40)
        print(f" - Updating world model: {iteration + 1}/{args.update_wm_epochs}")
        print("-"*40)

        if os.path.exists(DATASET_PATH):
            os.remove(DATASET_PATH)
            print(f"- Deleted old policy dataset: {DATASET_PATH}")

        if os.path.exists(AGENT_PATH):
            current_policy = AGENT_PATH
            policy_name = "Trained Agent"
        else:
            current_policy = None
            policy_name = "Random Policy"

        print(f"- Collecting {N_STEPS} steps with {policy_name}...")
        # env_rollout(ENV_NAME, N_STEPS, VAE, DATASET_PATH, policy_path=current_policy)
        env_rollout(
        ENV_NAME, 
        N_STEPS, 
        VAE, 
        DATASET_PATH, 
        policy_path=current_policy, 
        stack_frames=args.context_frames # <--- Linked to parser
        )
        
        print(f"- Training World Model for {DIT_EPOCHS} epochs...")
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
            patch_size=args.patch_size,
            weights_path=DIT_WEIGHTS_PATH
        )

        print('- Dreaming...')
        train_agent_in_dream(
            model_path=DIT_WEIGHTS_PATH,
            dataset_path=DATASET_PATH,
            agent_path=AGENT_PATH,
            steps=args.agent_steps,
            context_frames=args.context_frames,
            num_actions=NUM_ACTIONS,
            denoising_steps=args.denoising_steps,
            pixel_space=args.pixel_space,
            input_size=input_size,
            in_channels=in_channels,
            patch_size=args.patch_size,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads
        )



