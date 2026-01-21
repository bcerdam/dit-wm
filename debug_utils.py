import torch
import os
import h5py
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from vae import VAE
from atari_dataset import AtariH5Dataset
from rollout_gen import env_rollout, process_pixels_only, batch_encode, process_observations
import gymnasium as gym
from train_utils import DIT_CONFIGS, get_num_actions
from mod_dit import ModDiT


def inspect_dataset(file_path):
    with h5py.File(file_path, 'r') as f:
        print(f"\nInspecting: {file_path}")
        print("-" * 60)
        for key in f.keys():
            print(f"Key: {key:<12} | Shape: {str(f[key].shape):<18} | Type: {f[key].dtype}")
        print("-" * 60)


def create_video(dataset_path, output_filename, rollout_idx, vae_weights_path, latent_channel_dim,
                  latent_spatial_dim, observation_resolution, device, pixel_space, video_fps):
    
    vae = VAE(latent_channel_dim=latent_channel_dim, 
              latent_spatial_dim=latent_spatial_dim, 
              observation_resolution=observation_resolution).to(device)
    vae.load_state_dict(torch.load(vae_weights_path, map_location=device))
    vae.eval()

    with h5py.File(dataset_path, 'r') as f:
        dones = np.array(f['termination_status'])
        term_indices = np.where(dones)[0]
        
        if rollout_idx == 0:
            start_idx = 0
            end_idx = term_indices[0] + 1
        else:
            start_idx = term_indices[rollout_idx - 1] + 1
            end_idx = term_indices[rollout_idx] + 1

        print(f"Rendering Rollout {rollout_idx}: Steps {start_idx} to {end_idx}")
        raw_frames = f['observations'][start_idx:end_idx]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, video_fps, (observation_resolution, observation_resolution))
    
    for i in tqdm(range(len(raw_frames))):
        if pixel_space:
            frame = raw_frames[i]
            img_hwc = np.transpose(frame, (1, 2, 0)) 
            img_bgr = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)
            out.write(img_bgr)
        else:
            frame = raw_frames[i]
            frame_tensor = torch.from_numpy(frame).float().to(device).unsqueeze(0)

            with torch.no_grad():
                reconstruction = vae.decoder(frame_tensor)
            
            img_tensor = reconstruction.squeeze(0).cpu()
            img_hwc = img_tensor.permute(1, 2, 0).numpy()
            img_hwc = (img_hwc + 1.0) / 2.0 * 255.0
            
            img_uint8 = np.clip(img_hwc, 0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            out.write(img_bgr)

    out.release()
    print(f"Video saved to {output_filename}")

    
def visualize_denoising(model, dit_weights_path, vae_weights_path, output_filename, device, env_name, denoising_steps, pixel_space, 
                        data_shape, dtype, observation_resolution, video_fps, latent_channel_dim, latent_spatial_dim, context_length):
    
    temp_file = "temp_eval.h5"
    if os.path.exists(temp_file):
        os.remove(temp_file)

    if pixel_space == False:
        vae = VAE(latent_channel_dim=latent_channel_dim, 
                  latent_spatial_dim=latent_spatial_dim, 
                  observation_resolution=observation_resolution).to(device)
        vae.load_state_dict(torch.load(vae_weights_path, map_location=device))
        vae.eval()
    
    observations, actions, rewards, termination_status = env_rollout(
                                                                env_name=env_name,
                                                                n_steps=1,
                                                                dataset_path=temp_file, 
                                                                data_shape=data_shape, 
                                                                dtype=dtype,
                                                                resize_resolution=observation_resolution
                                                                )
    
    process_observations(n_steps=1,
                            all_episodes_observations=observations,
                            all_episodes_actions=actions,
                            all_episodes_rewards=rewards,
                            all_episodes_termination_status=termination_status,
                            pixel_space=pixel_space,
                            latent_channel_dim=latent_channel_dim,
                            latent_spatial_dim=latent_spatial_dim,
                            resize_resolution=observation_resolution,
                            vae_weights_path=vae_weights_path,
                            dataset_path=temp_file)
    
    
    model.load_state_dict(torch.load(dit_weights_path, map_location=device))
    model.to(device)
    model.eval()

    dataset = AtariH5Dataset(h5_path=temp_file, context_length=context_length)
    rand_idx = np.random.randint(0, len(dataset))
    target_clean, context, tgt_act, ctx_acts, _, _ = dataset[rand_idx]
    
    target_clean = target_clean.unsqueeze(0).to(device)
    context = context.unsqueeze(0).to(device)
    tgt_act = tgt_act.unsqueeze(0).to(device)
    ctx_acts = ctx_acts.unsqueeze(0).to(device)

    sigma_min = 0.002
    sigma_max = 80.0
    rho = 7.0
    
    latents = torch.randn_like(target_clean) * sigma_max
    
    step_indices = torch.arange(denoising_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (denoising_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]).float()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, video_fps, (observation_resolution, observation_resolution))

    print(f"Sampling reverse process (EDM, {denoising_steps} steps)...")
    with torch.no_grad():
        for i in tqdm(range(denoising_steps)):
            sigma_cur = t_steps[i]
            sigma_next = t_steps[i + 1]
            
            D_x = model(latents, sigma_cur.view(-1), context, tgt_act, ctx_acts)
            
            d_cur = (latents - D_x) / sigma_cur
            latents = latents + (sigma_next - sigma_cur) * d_cur

            viz_interval = max(1, denoising_steps // 50) 
            if i % viz_interval == 0 or i == denoising_steps - 1:
                
                if pixel_space:
                    # D_x = D_x.clamp(-1.0, 1.0)
                    img_tensor = (D_x[0].detach().cpu() + 1.0) / 2.0
                    img_tensor = torch.clamp(img_tensor, 0, 1)
                    img_np = img_tensor.permute(1, 2, 0).numpy()
                    img_np = (img_np * 255).astype(np.uint8)
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    out.write(img_bgr)
                else:
                    safe_latent = torch.nan_to_num(D_x, nan=0.0).clamp(-10, 10)
                    with torch.no_grad():
                        frame_pixels = vae.decoder(safe_latent).squeeze(0)
                    img_np = frame_pixels.permute(1, 2, 0).cpu().numpy()
                    img_uint8 = ((img_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
                    out.write(img_bgr)
                
                if i == denoising_steps - 1:
                    for _ in range(int(video_fps * 3)):
                        out.write(img_bgr)

    out.release()
    print(f"Video saved to {output_filename}")
    
    if env_name and os.path.exists(temp_file):
        os.remove(temp_file)


def edm_sampler(model, context, target_action, ctx_acts, device, input_size, in_channels, num_steps):
    sigma_min = 0.002
    sigma_max = 80.0
    rho = 7.0

    shape = (1, in_channels, input_size, input_size)
    latents = torch.randn(shape, device=device) * sigma_max
    
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)

    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]).float()
    
    for i in range(num_steps):
        sigma_cur = t_steps[i]
        sigma_next = t_steps[i + 1]

        with torch.no_grad():
            D_x = model(latents, sigma_cur.view(-1), context, target_action, ctx_acts)
        
        d_cur = (latents - D_x) / sigma_cur
        latents = latents + (sigma_next - sigma_cur) * d_cur
        
    return D_x


def dream_world(model, vae, env_name, output_filename, device, steps, pixel_space, context_frames,
                 num_steps, observation_resolution, num_actions, input_size, in_channels):
    print(f"- Dreaming {steps} frames, {num_steps} denoising steps)...")
    env = gym.make(env_name, render_mode='rgb_array')
    obs, _ = env.reset()
    
    real_frames = []
    real_actions = []
    for _ in range(context_frames):
        img = cv2.resize(obs, (observation_resolution, observation_resolution), interpolation=cv2.INTER_AREA)
        real_frames.append(img)
        action = env.action_space.sample()
        real_actions.append(action)
        obs, _, done, _, _ = env.step(action)
        if done: obs, _ = env.reset()

    env.close()

    if pixel_space:
        raw_pixels = process_pixels_only(real_frames)
        current_latents = (torch.tensor(raw_pixels).float() / 127.5) - 1.0
        current_latents = current_latents.to(device)
    else:
        current_latents = torch.tensor(batch_encode(real_frames, vae, device)).to(device)
        
    current_actions = torch.tensor(real_actions).to(device).long().unsqueeze(0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, 15.0, (observation_resolution, observation_resolution))

    for frame in real_frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
        out.write(frame_bgr)

    pbar = tqdm(range(steps))
    for i in pbar:
        recent_frames = current_latents[-context_frames:] 
        ctx_input = recent_frames.reshape(1, -1, recent_frames.shape[2], recent_frames.shape[3])
        ctx_act_input = current_actions[:, -context_frames:] 
        
        # next_action_val = np.random.randint(0, num_actions)
        next_action_val = 4
        tgt_act_input = torch.tensor([next_action_val], device=device).long()
        
        new_frame = edm_sampler(model=model, context=ctx_input, target_action=tgt_act_input, ctx_acts=ctx_act_input,
                                 input_size=input_size, in_channels=in_channels, num_steps=num_steps, device=device)
  
        current_latents = torch.cat([current_latents, new_frame], dim=0)
        current_actions = torch.cat([current_actions, tgt_act_input.unsqueeze(0)], dim=1)
        
        if len(current_latents) > context_frames + 5:
            current_latents = current_latents[-context_frames:]
            current_actions = current_actions[:, -context_frames:]

        if pixel_space:
            img_tensor = (new_frame[0].detach().cpu() + 1.0) / 2.0
            img_np = img_tensor.clamp(0, 1).permute(1, 2, 0).numpy()
            img_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        else:
            safe_latent = torch.nan_to_num(new_frame, nan=0.0).clamp(-10, 10)
            with torch.no_grad():
                reconstruction = vae.decoder(safe_latent)
            
            img_tensor = reconstruction.squeeze(0).cpu()
            img_hwc = img_tensor.permute(1, 2, 0).numpy()
            img_hwc = (img_hwc + 1.0) / 2.0 * 255.0
            
            img_uint8 = np.clip(img_hwc, 0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

        out.write(img_bgr)

    out.release()
    print(f"Dream saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Atari Dataset Utilities")
    parser.add_argument('mode', choices=['inspect', 'video', 'denoise', 'dream'], help='Select utility function to run')

    parser.add_argument('--env_name', type=str, default='ALE/Alien-v5', help='Gym Env ID for fresh/honest evaluation (e.g., ALE/Breakout-v5)')
    parser.add_argument('--observation_resolution', type=int, default=64, help='Image resolution of enviroment observations')
    parser.add_argument('--video_fps', type=int, default=60, help='Video FPS')

    parser.add_argument('--model', type=str, default='DiT-S', choices=list(DIT_CONFIGS.keys()), help='Standard DiT config')
    parser.add_argument('--patch_size', type=int, default=2, help='Patch size used in training (default 2 for latent, use 8 for pixel)')
    parser.add_argument('--hidden_size', type=int, default=384, help='Hidden dimension')
    parser.add_argument('--depth', type=int, default=6, help='Number of blocks')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of heads')
    parser.add_argument('--context_frames', type=int, default=4, help='Number of history frames')

    parser.add_argument('--latent_channel_dim', type=int, default=4, help='Channel dimension for latent VAE space')
    parser.add_argument('--latent_spatial_dim', type=int, default=32, help='Spatial dimension for latent VAE space')

    parser.add_argument('--denoising_steps', type=int, default=3, help='Number of sampling steps for EDM (default: 50)')

    parser.add_argument('--dataset_path', type=str, default='atari_dataset.h5', help='Path to .h5 dataset')
    parser.add_argument('--dit_weights_path', type=str, default='mod_dit.pt', help='Path to model weights (mod dit model)')
    parser.add_argument('--vae_weights_path', type=str, default='vae.pt', help='Path to the VAE weights')

    parser.add_argument('--output', type=str, default='output.mp4', help='Output video filename')
    parser.add_argument('--rollout_idx', type=int, default=0, help='Rollout index (video mode)')
    parser.add_argument('--max_frames', type=int, default=500, help='Number of frames to dream')
    parser.add_argument('--pixel_space', type=bool, default=False, help='Use if model is trained on pixels (64x64)')

    
    args = parser.parse_args()

    MODE = args.mode

    ENV_NAME = args.env_name
    NUM_ACTIONS = get_num_actions(ENV_NAME)
    OBSERVATION_RESOLUTION = args.observation_resolution
    VIDEO_FPS = args.video_fps
    DATASET_PATH = args.dataset_path

    PIXEL_SPACE = args.pixel_space

    MODEL = args.model
    PATCH_SIZE = args.patch_size
    HIDDEN_SIZE = args.hidden_size
    DEPTH = args.depth
    NUM_HEADS = args.num_heads
    CONTEXT_FRAMES = args.context_frames
    DIT_WEIGHTS_PATH = args.dit_weights_path
    VAE_WEIGHTS_PATH = args.vae_weights_path

    LATENT_CHANNEL_DIM = args.latent_channel_dim
    LATENT_SPATIAL_DIM = args.latent_spatial_dim

    MAX_FRAMES = args.max_frames
    DENOISING_STEPS = args.denoising_steps

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    OUTPUT_VIDEO_NAME = args.output
    ROLLOUT_IDX = args.rollout_idx


    if MODEL:
        config = DIT_CONFIGS[MODEL]
        HIDDEN_SIZE = config['hidden_size']
        DEPTH = config['depth']
        NUM_HEADS = config['num_heads']

    if PIXEL_SPACE:
        IN_CHANNELS = 3
        INPUT_SIZE = OBSERVATION_RESOLUTION
        DATA_SHAPE = (IN_CHANNELS, INPUT_SIZE, INPUT_SIZE)
        DTYPE = 'uint8'
        vae_required = False
    else:
        IN_CHANNELS = LATENT_CHANNEL_DIM
        INPUT_SIZE = LATENT_SPATIAL_DIM
        DATA_SHAPE = (IN_CHANNELS, INPUT_SIZE, INPUT_SIZE)
        DTYPE = 'float32'
        vae_required = True

    if MODE == 'inspect':
        inspect_dataset(DATASET_PATH)
        
    elif MODE == 'video':
        create_video(dataset_path=DATASET_PATH, 
                     output_filename=OUTPUT_VIDEO_NAME,
                     rollout_idx=ROLLOUT_IDX, 
                     vae_weights_path=VAE_WEIGHTS_PATH,
                     latent_channel_dim=LATENT_CHANNEL_DIM,
                     latent_spatial_dim=LATENT_SPATIAL_DIM,
                     observation_resolution=OBSERVATION_RESOLUTION,
                     device=DEVICE, 
                     pixel_space=PIXEL_SPACE,
                     video_fps=VIDEO_FPS)
        
    elif MODE == 'denoise':
        model = ModDiT(
            input_size=INPUT_SIZE,
            patch_size=PATCH_SIZE,
            in_channels=IN_CHANNELS, 
            context_frames=CONTEXT_FRAMES, 
            hidden_size=HIDDEN_SIZE, 
            depth=DEPTH, 
            num_heads=NUM_HEADS, 
            num_actions=NUM_ACTIONS,
            sigma_data=0.5
        )


        visualize_denoising(model=model,
                            dit_weights_path=DIT_WEIGHTS_PATH,
                            vae_weights_path=VAE_WEIGHTS_PATH,
                            output_filename=OUTPUT_VIDEO_NAME,
                            device=DEVICE,
                            env_name=ENV_NAME,
                            denoising_steps=DENOISING_STEPS,
                            pixel_space=PIXEL_SPACE,
                            data_shape=DATA_SHAPE,
                            dtype=DTYPE,
                            observation_resolution=OBSERVATION_RESOLUTION,
                            video_fps=VIDEO_FPS,
                            latent_channel_dim=LATENT_CHANNEL_DIM,
                            latent_spatial_dim=LATENT_SPATIAL_DIM,
                            context_length=CONTEXT_FRAMES)


    elif args.mode == 'dream':
        model = ModDiT(
            input_size=INPUT_SIZE,
            patch_size=PATCH_SIZE,
            in_channels=IN_CHANNELS, 
            context_frames=CONTEXT_FRAMES, 
            hidden_size=HIDDEN_SIZE, 
            depth=DEPTH, 
            num_heads=NUM_HEADS, 
            num_actions=NUM_ACTIONS,
            sigma_data=0.5
        )
        
        model.load_state_dict(torch.load(DIT_WEIGHTS_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        vae = VAE(latent_channel_dim=LATENT_CHANNEL_DIM, 
                  latent_spatial_dim=LATENT_SPATIAL_DIM, 
                  observation_resolution=OBSERVATION_RESOLUTION).to(DEVICE)
        vae.load_state_dict(torch.load(VAE_WEIGHTS_PATH, map_location=DEVICE))
        vae.eval()
        
        dream_world(model=model,
                    vae=vae,
                    env_name=ENV_NAME,
                    steps=MAX_FRAMES,
                    pixel_space=PIXEL_SPACE,
                    context_frames=CONTEXT_FRAMES,
                    num_steps=DENOISING_STEPS,
                    observation_resolution=OBSERVATION_RESOLUTION,
                    num_actions=NUM_ACTIONS,
                    input_size=INPUT_SIZE,
                    in_channels=IN_CHANNELS,
                    output_filename='output.mp4',
                    device=DEVICE
                    )
    