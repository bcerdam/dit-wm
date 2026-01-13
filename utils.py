import torch
import os
import h5py
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from diffusers.models import AutoencoderKL
from vae import decode_latent
from atari_dataset import AtariH5Dataset
from rollout_gen import env_rollout, process_pixels_only, batch_encode
import gymnasium as gym
import ale_py
from stable_baselines3 import PPO


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


def inspect_dataset(file_path):
    with h5py.File(file_path, 'r') as f:
        print(f"\nInspecting: {file_path}")
        print("=" * 60)
        
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
        

def create_video(dataset_path, output_filename, rollout_idx, vae, device='cuda', pixel_space=False):
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
        raw_frames = f['observations'][start_idx:end_idx]
        
        for i in tqdm(range(len(raw_frames))):

            if pixel_space == True:
                frame = raw_frames[i]
                img_hwc = np.transpose(frame, (1, 2, 0)) 
                
                if img_hwc.dtype != np.uint8:
                    img_hwc = img_hwc.astype(np.uint8)

                img_bgr = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)
                out.write(img_bgr)
            else:
                latent_batch = np.expand_dims(raw_frames[i], axis=0)
                img = decode_latent(latent_batch, vae, device=device)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                out.write(img_bgr)

        out.release()
        print(f"Video saved to {output_filename}")

    
def visualize_denoising(model, weights_path, output_filename='denoising.mp4', device='cuda', env_name=None, num_steps=50, pixel_space=False):
    vae = None
    if not pixel_space:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    temp_file = "temp_eval.h5"
    env_rollout(env_name, 1, vae, temp_file)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    dataset = AtariH5Dataset(temp_file)
    rand_idx = np.random.randint(0, len(dataset))
    target_clean, context, tgt_act, ctx_acts, reward, done, context_rewards, context_dones = dataset[rand_idx]
    
    
    target_clean = target_clean.unsqueeze(0).to(device)
    context = context.unsqueeze(0).to(device)
    tgt_act = tgt_act.unsqueeze(0).to(device)
    ctx_acts = ctx_acts.unsqueeze(0).to(device)

    reward = reward.unsqueeze(0).to(device)
    done = done.unsqueeze(0).to(device)
    context_rewards = context_rewards.unsqueeze(0).to(device)
    context_dones = context_dones.unsqueeze(0).to(device)

    cond_noise_level = torch.tensor([1e-5], device=device)

    sigma_min = 0.002
    sigma_max = 80.0
    rho = 7.0
    
    latents = torch.randn_like(target_clean) * sigma_max
    
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]).float()
    
    fps = 10.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (64, 64))

    print(f"Sampling reverse process (EDM, {num_steps} steps)...")
    
    with torch.no_grad():
        for i in tqdm(range(num_steps)):
            sigma_cur = t_steps[i]
            sigma_next = t_steps[i + 1]
            
            D_x, pred_reward, pred_done = model(
                latents, 
                sigma_cur.view(-1), 
                context, 
                cond_noise_level,
                tgt_act, 
                ctx_acts, 
                context_rewards, 
                context_dones
            )
            
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


def edm_sampler(model, context, target_action, ctx_acts, ctx_rews, ctx_dones, device, input_size, in_channels, num_steps=50):
    sigma_min = 0.002
    sigma_max = 80.0
    rho = 7.0

    shape = (1, in_channels, input_size, input_size)
    latents = torch.randn(shape, device=device) * sigma_max
    
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)

    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]).float()
    cond_noise_level = torch.tensor([1e-5], device=device)
    for i in range(num_steps):
        sigma_cur = t_steps[i]
        sigma_next = t_steps[i + 1]
        
        with torch.no_grad():
            D_x, pred_r, pred_d = model(
                latents, 
                sigma_cur.view(-1), 
                context, 
                cond_noise_level,
                target_action, 
                ctx_acts, 
                ctx_rews, 
                ctx_dones
            )
        
        d_cur = (latents - D_x) / sigma_cur
        latents = latents + (sigma_next - sigma_cur) * d_cur
        
    return D_x, pred_r, pred_d


def dream_world(model, vae, env_name, output_filename, device, steps=100, pixel_space=False, context_frames=4, num_steps=50, policy_path=None):

    if policy_path:
        agent = PPO.load(policy_path, device=device)
        policy_name = "PPO Agent"
    else:
        agent = None
        policy_name = "Random Policy"


    print(f"- Dreaming {steps} frames ({policy_name}, {num_steps} denoising steps)...")
    env = gym.make(env_name, render_mode='rgb_array')
    num_actions = env.action_space.n    
    obs, _ = env.reset()
    
    real_frames = []
    real_actions = []
    real_rewards = []
    real_dones = []
    for _ in range(context_frames):
        img = cv2.resize(obs, (64, 64))
        real_frames.append(img)
        
        if agent:
            action, _ = agent.predict(img, deterministic=True)
            action = int(action)
        else:
            # action = 3
            action = env.action_space.sample()
            
        real_actions.append(action)
        
        obs, reward, done, truncated, _ = env.step(action)
        real_rewards.append(reward)
        real_dones.append(done or truncated)
        if done: obs, _ = env.reset()

    env.close()

    if pixel_space:
        raw_pixels = process_pixels_only(real_frames)
        current_latents = (torch.tensor(raw_pixels).float() / 127.5) - 1.0
        current_latents = current_latents.to(device)
    else:
        current_latents = torch.tensor(batch_encode(real_frames, vae, device)).to(device)
        
    current_actions = torch.tensor(real_actions).to(device).long().unsqueeze(0)
    current_rewards = torch.tensor(real_rewards).to(device).float().view(1, -1, 1) # [1, T, 1]
    current_dones = torch.tensor(real_dones).to(device).long().view(1, -1)     # [1, T]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, 10.0, (64, 64))

    pbar = tqdm(range(steps))
    for i in pbar:
        recent_frames = current_latents[-context_frames:] 
        ctx_input = recent_frames.reshape(1, -1, recent_frames.shape[2], recent_frames.shape[3])
        ctx_act_input = current_actions[:, -context_frames:]
        ctx_rew_input = current_rewards[:, -context_frames:, :]
        ctx_done_input = current_dones[:, -context_frames:]
        
        if agent:
            last_frame_tensor = current_latents[-1]
            
            if pixel_space:
                obs_tensor = (last_frame_tensor + 1.0) / 2.0 * 255.0
                obs_np = obs_tensor.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                
                action, _ = agent.predict(obs_np, deterministic=True)
                next_action_val = int(action)
            else:
                with torch.no_grad():
                    lat = last_frame_tensor.unsqueeze(0) / 0.18215 
                    decoded = vae.decode(lat).sample
                    img = (decoded[0] / 2 + 0.5) * 255.0
                    img_np = img.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                    action, _ = agent.predict(img_np, deterministic=True)
            next_action_val = int(action)
        else:
            # for debugging purposes
            next_action_val = np.random.randint(0, num_actions)
            # next_action_val = 3

        tgt_act_input = torch.tensor([next_action_val], device=device).long()
        
        input_size = 64 if pixel_space else 8
        in_channels = 3 if pixel_space else 4
        
        new_frame, pred_r, pred_d = edm_sampler(
            model, ctx_input, tgt_act_input, ctx_act_input, ctx_rew_input, ctx_done_input,
            device, input_size, in_channels, num_steps=num_steps
        )
        
        current_latents = torch.cat([current_latents, new_frame], dim=0)
        current_actions = torch.cat([current_actions, tgt_act_input.unsqueeze(0)], dim=1)
        current_rewards = torch.cat([current_rewards, pred_r.view(1, 1, 1)], dim=1)
        if pred_d > 0.05:
            print(f'Termination: {pred_d}')

        if pred_r > 0.2:
            print(f'Reward: {pred_r}')

        is_done = (torch.sigmoid(pred_d) > 0.5).long()
        current_dones = torch.cat([current_dones, is_done.view(1, 1)], dim=1)
        
        if len(current_latents) > context_frames + 5:
            current_latents = current_latents[-context_frames:]
            current_actions = current_actions[:, -context_frames:]
            current_rewards = current_rewards[:, -context_frames:, :]
            current_dones = current_dones[:, -context_frames:]

        if pixel_space:
            img_tensor = (new_frame[0].detach().cpu() + 1.0) / 2.0
            img_np = img_tensor.clamp(0, 1).permute(1, 2, 0).numpy()
            img_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        else:
            safe_latent = torch.nan_to_num(new_frame, nan=0.0).clamp(-10, 10)
            img_rgb = decode_latent(safe_latent, vae, device)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        r_val = pred_r.item()
        d_val = torch.sigmoid(pred_d).item()
        text = f"R: {r_val:.2f} | D: {d_val:.2f}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.2
        thickness = 1
        color = (0, 255, 0)
        pos = (4, 12)

        cv2.putText(img_bgr, text, pos, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img_bgr, text, pos, font, scale, color, thickness, cv2.LINE_AA)
        out.write(img_bgr)

    out.release()
    print(f"Dream saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Atari Dataset Utilities")
    parser.add_argument('mode', choices=['inspect', 'video', 'denoise', 'dream'], help='Select utility function to run')

    parser.add_argument('--env_name', type=str, default='ALE/Breakout-v5', help='Gym Env ID for fresh/honest evaluation (e.g., ALE/Breakout-v5)')

    parser.add_argument('--model', type=str, default='DiT-S', choices=list(DIT_CONFIGS.keys()), help='Standard DiT config')
    parser.add_argument('--context_frames', type=int, default=4, help='Number of history frames')
    parser.add_argument('--patch_size', type=int, default=2, help='Patch size used in training (default 2 for latent, use 8 for pixel)')
    parser.add_argument('--hidden_size', type=int, default=384, help='Hidden dimension')
    parser.add_argument('--depth', type=int, default=6, help='Number of blocks')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of heads')

    parser.add_argument('--denoising_steps', type=int, default=5, help='Number of sampling steps for EDM (default: 50)')

    parser.add_argument('--dataset_path', type=str, default='atari_dataset.h5', help='Path to .h5 dataset')

    parser.add_argument('--weights_path', type=str, default='mod_dit.pt', help='Path to model weights (denoise mode)')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video filename')
    parser.add_argument('--rollout_idx', type=int, default=0, help='Rollout index (video mode)')
    parser.add_argument('--max_frames', type=int, default=1000, help='Number of frames to dream')
    parser.add_argument('--pixel_space', type=bool, default=False, help='Use if model is trained on pixels (64x64)')

    parser.add_argument('--policy_path', type=str, default='ppo_dream_agent.zip', help='Path to trained PPO agent for dreaming')

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_ACTIONS = get_num_actions(args.env_name)

    if args.pixel_space:
        in_channels = 3
        input_size = 64
        vae = None
    else:
        in_channels = 4
        input_size = 8
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    if args.model:
        config = DIT_CONFIGS[args.model]
        print(f"- Using standard configuration for {args.model}")
        args.hidden_size = config['hidden_size']
        args.depth = config['depth']
        args.num_heads = config['num_heads']

    if args.mode == 'inspect':
        inspect_dataset(args.dataset_path)
        
    elif args.mode == 'video':
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
        create_video(args.dataset_path, args.output, args.rollout_idx, vae, device, args.pixel_space)
        
    elif args.mode == 'denoise':
        from mod_dit import ModDiT
        
        model = ModDiT(
            in_channels=in_channels, 
            context_frames=args.context_frames, 
            num_actions=NUM_ACTIONS,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            sigma_data=0.5,
            input_size=input_size,
            patch_size=args.patch_size
        )

        visualize_denoising(
            model, 
            args.weights_path, 
            args.output, 
            device=device, 
            env_name=args.env_name, 
            num_steps=args.denoising_steps,
            pixel_space=args.pixel_space
        )


    elif args.mode == 'dream':
        from mod_dit import ModDiT
        model = ModDiT(
            in_channels=in_channels, 
            context_frames=args.context_frames, 
            num_actions=NUM_ACTIONS,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            sigma_data=0.5,
            input_size=input_size,
            patch_size=args.patch_size
        )
        
        model.load_state_dict(torch.load(args.weights_path, map_location=device))
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
            num_steps=args.denoising_steps,
            context_frames=args.context_frames,
            policy_path=args.policy_path
        )
    