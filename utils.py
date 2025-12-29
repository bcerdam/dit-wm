import torch
import os
import h5py
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from diffusers.models import AutoencoderKL
from vae import decode_latent
from atari_dataset import AtariH5Dataset
from rollout_gen_p import env_rollout

DIT_CONFIGS = {
    'DiT-S': {'hidden_size': 384, 'depth': 12, 'num_heads': 6},
    'DiT-B': {'hidden_size': 768, 'depth': 12, 'num_heads': 12},
    'DiT-L': {'hidden_size': 1024, 'depth': 24, 'num_heads': 16},
    'DiT-XL': {'hidden_size': 1152, 'depth': 28, 'num_heads': 16},
}


'''
plot_atari_frame(): Plots observation from atari game.

obs: Observation from atari game.
'''
def plot_atari_frame(obs):
    img = Image.fromarray(obs)
    img.save('test.jpeg')


'''
inspect_dataset(): Utility function that checks common features of atari dataset. Basically a sanity check.

file_path: Path of atari-100k dataset.
'''
def inspect_dataset(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    with h5py.File(file_path, 'r') as f:
        print(f"\nInspecting: {file_path}")
        print("=" * 60)
        
        if len(f.keys()) == 0:
            print("(!) File is empty.")
            return

        for key in f.keys():
            print(f"Key: {key:<12} | Shape: {str(f[key].shape):<18} | Type: {f[key].dtype}")

        if 'terminated' in f:
            dones = np.array(f['terminated'])
            total_steps = len(dones)
            total_episodes = np.sum(dones)
            
            print("-" * 60)
            print(f"Total Steps:    {total_steps}")
            print(f"Total Episodes: {total_episodes}")
            
            if total_episodes > 0:
                term_indices = np.where(dones)[0]
                
                boundaries = np.concatenate(([-1], term_indices))
                lengths = np.diff(boundaries)
                
                print(f"Avg Length:     {np.mean(lengths):.2f}")
                print(f"Min Length:     {np.min(lengths)}")
                print(f"Max Length:     {np.max(lengths)}")
                print(f"Lengths (first 10): {lengths[:10].tolist()}")
        
        print("=" * 60)
        

'''
create_video(): Decodes latents from HDF5 and saves an .mp4 video.
'''
def create_video(dataset_path, output_filename, rollout_idx, vae, device='cuda'):
    with h5py.File(dataset_path, 'r') as f:
        dones = np.array(f['terminated'])
        term_indices = np.where(dones)[0]
        
        if rollout_idx == 0:
            start_idx = 0
            end_idx = term_indices[0] + 1
        else:
            if rollout_idx >= len(term_indices):
                print(f"Error: Rollout {rollout_idx} not found. Total episodes: {len(term_indices)}")
                return
            start_idx = term_indices[rollout_idx - 1] + 1
            end_idx = term_indices[rollout_idx] + 1

        print(f"Rendering Rollout {rollout_idx}: Steps {start_idx} to {end_idx} ({end_idx - start_idx} frames)")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, 30.0, (64, 64))

        latents = f['latents'][start_idx:end_idx]
        
        for i in tqdm(range(len(latents))):
            latent_batch = np.expand_dims(latents[i], axis=0)
            
            img = decode_latent(latent_batch, vae, device=device)
            
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            out.write(img_bgr)

        out.release()
        print(f"Video saved to {output_filename}")


def visualize_raw_agent(env_name, output_filename='raw_agent.mp4'):
    try:
        import gymnasium as gym
    except ImportError:
        import gym

    temp_path = "temp_raw_rollout.h5"
    print(f"(!) Collecting RAW observations from {env_name} (Resized to 64x64)...")
    
    env = gym.make(env_name, render_mode='rgb_array')
    obs, _ = env.reset()
    
    frames = []
    actions = []
    terminated = []
    
    frame_64 = cv2.resize(obs, (64, 64), interpolation=cv2.INTER_AREA)
    frames.append(frame_64)
    terminated.append(False)
    
    done = False
    trunc = False
    
    while not (done or trunc):
        action = env.action_space.sample()
        obs, reward, done, trunc, _ = env.step(action)
        
        frame_64 = cv2.resize(obs, (64, 64), interpolation=cv2.INTER_AREA)
        frames.append(frame_64)
        actions.append(action)
        terminated.append(done or trunc)
        
        if len(frames) >= 500:
            break
            
    env.close()

    print(f"Saving temporary dataset to {temp_path}...")
    with h5py.File(temp_path, 'w') as f:
        f.create_dataset('observations', data=np.array(frames), compression='gzip')
        f.create_dataset('actions', data=np.array(actions))
        f.create_dataset('terminated', data=np.array(terminated))

    print(f"Rendering raw video to {output_filename}...")
    
    with h5py.File(temp_path, 'r') as f:
        data = f['observations'][:]
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, 30.0, (64, 64))
    
    for img_rgb in tqdm(data):
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        out.write(img_bgr)
        
    out.release()
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
    


'''
visualize_denoising(): Decodes latents from HDF5 and saves an .mp4 video.
'''
def visualize_denoising(model, weights_path, dataset_path, output_filename='denoising.mp4', device='cuda', env_name=None):
    if not os.path.exists(weights_path):
        print(f"Error: Weights file '{weights_path}' not found.")
        return

    print(f"Loading VAE on {device}...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    temp_file = "temp_honest_eval.h5"
    if env_name:
        print(f"(!) Generating FRESH rollout from {env_name} to ensure unseen data...")
        if os.path.exists(temp_file): os.remove(temp_file)
        # Generate 1 fresh episode
        env_rollout(env_name, 1, vae, temp_file)
        dataset_path = temp_file

    print(f"Creating denoising video from {weights_path} using {dataset_path}...")
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    dataset = AtariH5Dataset(dataset_path)
    if len(dataset) == 0:
        print("Error: Dataset is empty.")
        return

    rand_idx = np.random.randint(0, len(dataset))
    target_clean, context, tgt_act, ctx_acts = dataset[rand_idx]
    
    target_clean = target_clean.unsqueeze(0).to(device)
    context = context.unsqueeze(0).to(device)
    tgt_act = tgt_act.unsqueeze(0).to(device)
    ctx_acts = ctx_acts.unsqueeze(0).to(device)

    xt = torch.randn_like(target_clean)
    
    fps = 10.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (64, 64))

    print("Sampling reverse process...")
    num_steps = 1000
    
    with torch.no_grad():
        for t_idx in tqdm(reversed(range(num_steps)), total=num_steps):
            t = torch.tensor([t_idx], device=device).long()
            
            model_input = torch.cat([xt, context], dim=1)
            noise_pred = model(model_input, t, tgt_act, ctx_acts)
            
            if t_idx > 0:
                alpha_bar_t = 1 - t_idx / 1000.0
                alpha_bar_prev = 1 - (t_idx - 1) / 1000.0
                
                beta_t = 1 - (alpha_bar_t / alpha_bar_prev)
                alpha_t = 1.0 - beta_t
                sigma_t = np.sqrt(beta_t)
                
                z = torch.randn_like(xt)
                xt = (1 / np.sqrt(alpha_t)) * (xt - (beta_t / np.sqrt(1 - alpha_bar_t)) * noise_pred) + sigma_t * z

            if t_idx % 20 == 0 or t_idx == 0:
                frame_pixels = decode_latent(xt, vae, device=device)
                img_bgr = cv2.cvtColor(frame_pixels, cv2.COLOR_RGB2BGR)
                out.write(img_bgr)
                
                if t_idx == 0:
                    for _ in range(int(fps * 3)):
                        out.write(img_bgr)

    out.release()
    print(f"Video saved to {output_filename}")

    if env_name and os.path.exists(temp_file):
        os.remove(temp_file)
        print("(!) Cleaned up temporary evaluation dataset.")

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-torch.log(torch.tensor(max_period)) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2: embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def unpatchify(x, channels):
    p = 2
    h = w = int(x.shape[1] ** 0.5)
    x = x.reshape(shape=(x.shape[0], h, w, p, p, channels))
    x = torch.einsum('nhwpqc->nchpwq', x)
    return x.reshape(shape=(x.shape[0], channels, h * p, h * p))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Atari Dataset Utilities")
    parser.add_argument('mode', choices=['inspect', 'video', 'denoise', 'video_raw'], help='Select utility function to run')
    
    parser.add_argument('--path', type=str, default='atari_dataset.h5', help='Path to .h5 dataset')
    parser.add_argument('--weights', type=str, default='dit_wm.pt', help='Path to model weights (denoise mode)')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video filename')
    parser.add_argument('--rollout_idx', type=int, default=0, help='Rollout index (video mode)')
    
    parser.add_argument('--model', type=str, default=None, choices=list(DIT_CONFIGS.keys()), help='Standard DiT config')
    parser.add_argument('--hidden_size', type=int, default=384, help='Hidden dimension')
    parser.add_argument('--depth', type=int, default=6, help='Number of blocks')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of heads')

    parser.add_argument('--env_name', type=str, default='ALE/Breakout-v5', help='Gym Env ID for fresh/honest evaluation (e.g., ALE/Breakout-v5)')
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model:
        config = DIT_CONFIGS[args.model]
        print(f"(!) Using standard configuration for {args.model}")
        args.hidden_size = config['hidden_size']
        args.depth = config['depth']
        args.num_heads = config['num_heads']

    if args.mode == 'inspect':
        inspect_dataset(args.path)
        
    elif args.mode == 'video':
        print(f"Loading VAE on {device}...")
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
        create_video(args.path, args.output, args.rollout_idx, vae, device)
        
    elif args.mode == 'denoise':
        try:
            from mod_dit import DiT_WM
        except ImportError:
            print("Error: Could not import DiT_WM from mod_dit.py.")
            exit()
            
        print(f"Initializing Model (H={args.hidden_size}, D={args.depth})...")
        
        model = DiT_WM(
            in_channels=4, 
            context_frames=4, 
            num_actions=18,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads
        )
        
        visualize_denoising(model, args.weights, args.path, args.output, device=device, env_name=args.env_name)

    elif args.mode == 'video_raw':
        if not args.env_name:
            print("Error: --env_name is required for raw video generation (e.g. ALE/Breakout-v5)")
            exit()
        visualize_raw_agent(args.env_name, args.output)

    