import argparse
import os
from diffusers.models import AutoencoderKL

from rollout_gen import env_rollout

### nsnm ###
from mod_dit import train_mod_dit
from utils import get_num_actions, DIT_CONFIGS
### nsnm ###


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DiT-WM")
    
    parser.add_argument('--env_name', type=str, default='ALE/Breakout-v5', help='Atari environment ID')
    parser.add_argument('--observation_resolution', type=int, default=64, help='Image resolution of enviroment observations')
    parser.add_argument('--n_steps', type=int, default=1000, help='Total environment steps to collect')

    parser.add_argument('--val_split', type=float, default=0.2, help='Ratio of data used for validation (e.g., 0.1 for 10%)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for DiT training')
    parser.add_argument('--vae_n_epochs', type=int, default=50, help='Training epochs for VAE')
    parser.add_argument('--dit_n_epochs', type=int, default=50, help='Training epochs for Dynamics Model')

    parser.add_argument('--model', type=str, default='DiT-S', choices=list(DIT_CONFIGS.keys()), help='Standard DiT config')
    parser.add_argument('--patch_size', type=int, default=2, help='Size of image patches (use 2 for Latent, 4 or 8 for Pixel)')
    parser.add_argument('--hidden_size', type=int, default=384, help='Transformer embedding dimension')
    parser.add_argument('--depth', type=int, default=6, help='Number of DiT blocks')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--context_frames', type=int, default=4, help='Number of history frames')

    parser.add_argument('--latent_channel_dim', type=int, default=4, help='Channel dimension for latent VAE space')
    parser.add_argument('--latent_spatial_dim', type=int, default=32, help='Spatial dimension for latent VAE space')

    parser.add_argument('--dataset_path', type=str, default='data/atari_dataset.h5', help='Path to HDF5 dataset')
    parser.add_argument('--weights_path', type=str, default='weights/mod_dit.pt', help='Path to model weights (denoise mode)')
    
    parser.add_argument('--delete_dataset', action='store_true', default=True, help='If set, deletes existing dataset')
    parser.add_argument('--keep_dataset', action='store_false', dest='delete_dataset', help='Keep existing dataset')

    parser.add_argument('--delete_dit_weights', action='store_true', default=True, help='If set, deletes existing DiT weights')
    parser.add_argument('--keep_dit_weights', action='store_false', dest='delete_dit_weights', help='Keep existing DiT weights')
    
    parser.add_argument('--pixel_space', type=bool, default=False, help='If set, trains on 64x64 RGB pixels instead of VAE latents')
    
    args = parser.parse_args()

    ENV_NAME = args.env_name
    NUM_ACTIONS = get_num_actions(ENV_NAME)
    OBSERVATION_RESOLUTION = args.observation_resolution
    N_STEPS = args.n_steps
    DATASET_PATH = args.dataset_path

    PIXEL_SPACE = args.pixel_space

    VAL_SPLIT = args.val_split
    BATCH_SIZE = args.batch_size
    VAE_EPOCHS = args.vae_n_epochs
    DIT_EPOCHS = args.dit_n_epochs

    MODEL = args.model
    PATCH_SIZE = args.patch_size
    HIDDEN_SIZE = args.hidden_size
    DEPTH = args.depth
    NUM_HEADS = args.num_heads
    CONTEXT_FRAMES = args.context_frames
    DIT_WEIGHTS_PATH = args.weights_path

    LATENT_CHANNEL_DIM = args.latent_channel_dim
    LATENT_SPATIAL_DIM = args.latent_spatial_dim

    DELETE_DATASET = args.delete_dataset
    DELETE_DIT_WEIGHTS = args.delete_dit_weights

    if MODEL:
        config = DIT_CONFIGS[MODEL]
        HIDDEN_SIZE = config['hidden_size']
        DEPTH = config['depth']
        NUM_HEADS = config['num_heads']

    if DELETE_DATASET and os.path.exists(DATASET_PATH):
        os.remove(DATASET_PATH)

    if DELETE_DIT_WEIGHTS and os.path.exists(DIT_WEIGHTS_PATH):
        os.remove(DIT_WEIGHTS_PATH)

    if PIXEL_SPACE:
        VAE = None 
        IN_CHANNELS = 3
        INPUT_SIZE = OBSERVATION_RESOLUTION
        DATA_SHAPE = (IN_CHANNELS, INPUT_SIZE, INPUT_SIZE)
        DTYPE = 'uint8'
    else:
        VAE = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to('cuda')
        IN_CHANNELS = 4
        INPUT_SIZE = 8
        DATA_SHAPE = (IN_CHANNELS, INPUT_SIZE, INPUT_SIZE)
        DTYPE = 'float32'

    env_rollout(env_name=ENV_NAME,
                n_steps=N_STEPS,
                vae=VAE,
                dataset_path=DATASET_PATH, 
                pixel_space=PIXEL_SPACE, 
                data_shape=DATA_SHAPE, 
                dtype=DTYPE,
                resize_resolution=OBSERVATION_RESOLUTION,
                val_split=VAL_SPLIT,
                batch_size=BATCH_SIZE,
                vae_epochs=VAE_EPOCHS,
                latent_channel_dim=LATENT_CHANNEL_DIM,
                latent_spatial_dim=LATENT_SPATIAL_DIM)

    # # ### nsnm ###
    # train_mod_dit(
    #     dataset_path=DATASET_PATH, 
    #     num_actions=NUM_ACTIONS,
    #     epochs=DIT_EPOCHS, 
    #     batch_size=BATCH_SIZE, 
    #     val_split=VAL_SPLIT,
    #     in_channels=IN_CHANNELS,
    #     context_frames=CONTEXT_FRAMES,
    #     hidden_size=HIDDEN_SIZE,
    #     depth=DEPTH,
    #     num_heads=NUM_HEADS,
    #     input_size=INPUT_SIZE, 
    #     patch_size=PATCH_SIZE
    # )
    # # ### nsnm ###



