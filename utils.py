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
from rollout_gen import env_rollout
import gymnasium as gym
import ale_py


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


def create_video_pixel(dataset_path, output_filename, rollout_idx):
    import h5py
    import cv2
    import numpy as np
    from tqdm import tqdm

    with h5py.File(dataset_path, 'r') as f:
        # 1. Locate the specific episode
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

        # 2. Setup Video Writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, 30.0, (64, 64))

        # 3. Read raw pixels (No VAE needed)
        # Note: Even in pixel space, the dataset key is often named 'latents' in your code
        raw_frames = f['latents'][start_idx:end_idx]
        
        for i in tqdm(range(len(raw_frames))):
            frame = raw_frames[i] # Shape is [3, 64, 64]
            
            # HDF5 stores [Channels, Height, Width], OpenCV needs [Height, Width, Channels]
            img_hwc = np.transpose(frame, (1, 2, 0)) 
            
            # Ensure it is uint8 (0-255)
            if img_hwc.dtype != np.uint8:
                img_hwc = img_hwc.astype(np.uint8)

            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)
            
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
    

def visualize_denoising(model, weights_path, dataset_path, output_filename='denoising.mp4', device='cuda', env_name=None, num_steps=50, pixel_space=False):
    if not os.path.exists(weights_path):
        print(f"Error: Weights file '{weights_path}' not found.")
        return

    vae = None
    if not pixel_space:
        print(f"Loading VAE on {device}...")
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    else:
        print("(!) Visualizing in PIXEL SPACE (No VAE).")

    temp_file = "temp_honest_eval.h5"
    if env_name:
        print(f"(!) Generating FRESH rollout from {env_name} to ensure unseen data...")
        if os.path.exists(temp_file): os.remove(temp_file)
        env_rollout(env_name, 1, vae, temp_file, policy_path='ppo_agent.zip')
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
    # Note: dataset now returns 6 items (added reward/done), but we only need the first 4 here
    target_clean, context, tgt_act, ctx_acts, _, _ = dataset[rand_idx]
    
    target_clean = target_clean.unsqueeze(0).to(device)
    context = context.unsqueeze(0).to(device)
    tgt_act = tgt_act.unsqueeze(0).to(device)
    ctx_acts = ctx_acts.unsqueeze(0).to(device)

    sigma_min = 0.002
    sigma_max = 80.0
    rho = 7.0
    
    latents = torch.randn_like(target_clean) * sigma_max
    
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]).float()

    cond_noise_level = torch.tensor([1e-5], device=device)
    
    fps = 10.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (64, 64))

    print(f"Sampling reverse process (EDM, {num_steps} steps)...")
    
    with torch.no_grad():
        for i in tqdm(range(num_steps)):
            sigma_cur = t_steps[i]
            sigma_next = t_steps[i + 1]
            
            # --- FIX: Unpack tuple (D_x, reward, done) ---
            D_x, _, _ = model(latents, sigma_cur.view(-1), context, cond_noise_level, tgt_act, ctx_acts)
            
            d_cur = (latents - D_x) / sigma_cur
            latents = latents + (sigma_next - sigma_cur) * d_cur

            viz_interval = max(1, num_steps // 50) 
            if i % viz_interval == 0 or i == num_steps - 1:
                
                if pixel_space:
                    img_tensor = (D_x[0].detach().cpu() + 1.0) / 2.0
                    img_tensor = torch.clamp(img_tensor, 0, 1)
                    img_np = img_tensor.permute(1, 2, 0).numpy()
                    img_np = (img_np * 255).astype(np.uint8)
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    out.write(img_bgr)
                else:
                    safe_latent = torch.nan_to_num(D_x, nan=0.0).clamp(-10, 10)
                    frame_pixels = decode_latent(safe_latent, vae, device=device)
                    img_bgr = cv2.cvtColor(frame_pixels, cv2.COLOR_RGB2BGR)
                    out.write(img_bgr)
                
                if i == num_steps - 1:
                    for _ in range(int(fps * 3)):
                        out.write(img_bgr)

    out.release()
    print(f"Video saved to {output_filename}")
    
    if env_name and os.path.exists(temp_file):
        os.remove(temp_file)


def unpatchify(x, channels):
    patch_dim = x.shape[-1]
    p = int((patch_dim // channels) ** 0.5)
    h = w = int(x.shape[1] ** 0.5)
    x = x.reshape(shape=(x.shape[0], h, w, p, p, channels))
    x = torch.einsum('nhwpqc->nchpwq', x)
    return x.reshape(shape=(x.shape[0], channels, h * p, h * p))


"""Generates a single frame using EDM sampling."""
def edm_sampler(model, context, target_action, ctx_acts, device, input_size, in_channels, num_steps=50):
    sigma_min = 0.002
    sigma_max = 80.0
    rho = 7.0

    shape = (1, in_channels, input_size, input_size)
    latents = torch.randn(shape, device=device) * sigma_max
    
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)

    # Double check if its the right formula
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]).float()
    
    cond_noise_level = torch.tensor([1e-5], device=device)

    D_x = None
    for i in range(num_steps):
        sigma_cur = t_steps[i]
        sigma_next = t_steps[i + 1]
        
        with torch.no_grad():
            D_x, pred_r, pred_d = model(latents, sigma_cur.view(-1), context, cond_noise_level, target_action, ctx_acts)
        
        d_cur = (latents - D_x) / sigma_cur
        latents = latents + (sigma_next - sigma_cur) * d_cur
        
    return D_x, pred_r, pred_d

def dream_world(model, vae, env_name, output_filename, device, steps=100, pixel_space=False, context_frames=4, num_steps=50, policy_path=None):
    """
    Autoregressive hallucination loop with optional Agent Control.
    """
    import gymnasium as gym
    import ale_py
    from rollout_gen import process_pixels_only, batch_encode
    from stable_baselines3 import PPO # Requires pip install stable-baselines3

    # Load Agent if path provided
    agent = None
    policy_name = "Random Policy"
    if policy_path:
        try:
            print(f"(!) Loading PPO Agent from {policy_path}...")
            agent = PPO.load(policy_path)
            policy_name = "PPO Agent"
        except Exception as e:
            print(f"(!) Failed to load policy: {e}. Defaulting to Random.")

    print(f"(!) Dreaming {steps} frames ({policy_name}, {num_steps} denoising steps)...")
    
    # --- 1. WARMUP (REAL ENV) ---
    env = gym.make(env_name, render_mode='rgb_array')
    num_actions = env.action_space.n
    print(f"(!) Detected {num_actions} valid actions for {env_name}.")
    
    obs, _ = env.reset()
    
    real_frames = []
    real_actions = []
    
    # Warmup Loop
    for _ in range(context_frames):
        img = cv2.resize(obs, (64, 64))
        real_frames.append(img)
        
        # If we have an agent, let it pick the warmup actions too!
        if agent:
            # FIX: Pass 'img' (which is resized to 64x64)
            action, _ = agent.predict(img, deterministic=True)
            action = int(action) # <--- FORCE PYTHON INT
        else:
            action = env.action_space.sample()
            
        real_actions.append(action)
        
        obs, _, done, _, _ = env.step(action)
        if done: obs, _ = env.reset()

    env.close()

    # --- 2. INITIALIZATION ---
    if pixel_space:
        raw_pixels = process_pixels_only(real_frames)
        current_latents = (torch.tensor(raw_pixels).float() / 127.5) - 1.0
        current_latents = current_latents.to(device)
    else:
        current_latents = torch.tensor(batch_encode(real_frames, vae, device)).to(device)
        
    current_actions = torch.tensor(real_actions).to(device).long().unsqueeze(0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, 10.0, (64, 64))

    # --- 3. AUTOREGRESSIVE LOOP ---
    pbar = tqdm(range(steps))
    for i in pbar:
        # A. PREPARE CONTEXT
        recent_frames = current_latents[-context_frames:] 
        ctx_input = recent_frames.reshape(1, -1, recent_frames.shape[2], recent_frames.shape[3])
        ctx_act_input = current_actions[:, -context_frames:] 
        
        # B. PICK NEXT ACTION
        if agent:
            # 1. Get current "observation" (Last frame in buffer)
            # Shape is [C, H, W] in range [-1, 1]
            last_frame_tensor = current_latents[-1]
            
            if pixel_space:
                # Un-normalize to [0, 255]
                obs_tensor = (last_frame_tensor + 1.0) / 2.0 * 255.0
                obs_np = obs_tensor.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy() # [H, W, C]
                
                # Predict
                action, _ = agent.predict(obs_np, deterministic=True)
                next_action_val = int(action)
            else:
                # If latent space, we assume the agent was trained on Latents? 
                # Or we must decode to pixels for a pixel-based agent.
                # Assuming pixel-based agent -> Decode necessary (slow)
                # skipping implementation for latent-agent for brevity unless requested
                next_action_val = np.random.randint(0, num_actions)
        else:
            next_action_val = np.random.randint(0, num_actions)

        # Create Tensor Input
        tgt_act_input = torch.tensor([next_action_val], device=device).long()
        
        # C. PREDICT NEXT FRAME
        input_size = 64 if pixel_space else 8
        in_channels = 3 if pixel_space else 4
        
        new_frame, _, _ = edm_sampler(
            model, ctx_input, tgt_act_input, ctx_act_input, device, 
            input_size, in_channels, num_steps=num_steps
        )
        
        # D. UPDATE HISTORY
        current_latents = torch.cat([current_latents, new_frame], dim=0)
        current_actions = torch.cat([current_actions, tgt_act_input.unsqueeze(0)], dim=1)
        
        if len(current_latents) > context_frames + 5:
            current_latents = current_latents[-context_frames:]
            current_actions = current_actions[:, -context_frames:]

        # E. SAVE VIDEO
        if pixel_space:
            img_tensor = (new_frame[0].detach().cpu() + 1.0) / 2.0
            img_np = img_tensor.clamp(0, 1).permute(1, 2, 0).numpy()
            img_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        else:
            safe_latent = torch.nan_to_num(new_frame, nan=0.0).clamp(-10, 10)
            img_rgb = decode_latent(safe_latent, vae, device)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        out.write(img_bgr)

    out.release()
    print(f"Dream saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Atari Dataset Utilities")
    parser.add_argument('mode', choices=['inspect', 'video', 'denoise', 'video_raw', 'dream'], help='Select utility function to run')
    
    parser.add_argument('--path', type=str, default='atari_dataset.h5', help='Path to .h5 dataset')
    parser.add_argument('--weights', type=str, default='dit_wm.pt', help='Path to model weights (denoise mode)')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video filename')
    parser.add_argument('--rollout_idx', type=int, default=0, help='Rollout index (video mode)')
    
    parser.add_argument('--model', type=str, default='DiT-S', choices=list(DIT_CONFIGS.keys()), help='Standard DiT config')
    parser.add_argument('--hidden_size', type=int, default=384, help='Hidden dimension')
    parser.add_argument('--depth', type=int, default=6, help='Number of blocks')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of heads')

    parser.add_argument('--env_name', type=str, default='ALE/Breakout-v5', help='Gym Env ID for fresh/honest evaluation (e.g., ALE/Breakout-v5)')
    
    parser.add_argument('--num_steps', type=int, default=3, help='Number of sampling steps for EDM (default: 50)')
    parser.add_argument('--max_frames', type=int, default=100, help='Number of frames to dream')
    parser.add_argument('--pixel_space', type=bool, default=True, help='Use if model is trained on pixels (64x64)')
    parser.add_argument('--patch_size', type=int, default=8, help='Patch size used in training (default 2 for latent, use 8 for pixel)')

    parser.add_argument('--policy', type=str, default='ppo_agent.zip', help='Path to PPO agent (e.g. ppo_agent.zip)')

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
        # print(f"Loading VAE on {device}...")
        # vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
        # create_video(args.path, args.output, args.rollout_idx, vae, device)
        create_video_pixel(args.path, args.output, args.rollout_idx)
        
    elif args.mode == 'denoise':
        try:
            from mod_dit import DiT_WM
        except ImportError:
            print("Error: Could not import DiT_WM from mod_dit.py.")
            exit()
            
        if args.pixel_space:
            in_channels = 3
            input_size = 64
        else:
            in_channels = 4
            input_size = 8
            
        print(f"Initializing Model (H={args.hidden_size}, D={args.depth}, Patch={args.patch_size}, Pixel={args.pixel_space})...")
        
        model = DiT_WM(
            in_channels=in_channels, 
            context_frames=4, 
            num_actions=4,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            sigma_data=0.5,
            input_size=input_size,
            patch_size=args.patch_size
        )

        visualize_denoising(
            model, 
            args.weights, 
            args.path, 
            args.output, 
            device=device, 
            env_name=args.env_name, 
            num_steps=args.num_steps,
            pixel_space=args.pixel_space
        )

    elif args.mode == 'video_raw':
        if not args.env_name:
            print("Error: --env_name is required for raw video generation (e.g. ALE/Breakout-v5)")
            exit()
        visualize_raw_agent(args.env_name, args.output)


    elif args.mode == 'dream':
        try:
            from mod_dit import DiT_WM
        except ImportError:
            exit()

        if args.pixel_space:
            in_channels, input_size = 3, 64
            vae = None
        else:
            in_channels, input_size = 4, 8
            vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

        model = DiT_WM(
            in_channels=in_channels, 
            context_frames=4, 
            num_actions=4,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            sigma_data=0.5,
            input_size=input_size,
            patch_size=args.patch_size
        )
        
        model.load_state_dict(torch.load(args.weights, map_location=device))
        model.to(device)
        model.eval()
        
        dream_world(
            model, 
            vae, 
            args.env_name, 
            args.output, 
            device, 
            steps=args.max_frames, 
            pixel_space=args.pixel_space, 
            num_steps=args.num_steps,
            policy_path=args.policy
        )
    