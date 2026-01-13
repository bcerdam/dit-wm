import os
from stable_baselines3 import PPO
from dream_env import DreamEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecFrameStack

def train_agent_in_dream(model_path, dataset_path, agent_path, steps=10000, 
                         context_frames=64, num_actions=4, device='cuda',
                         denoising_steps=5, pixel_space=False,
                         input_size=8, in_channels=4, patch_size=2,
                         hidden_size=384, depth=6, num_heads=6):
    
    env = DreamEnv(
        model_path=model_path, 
        dataset_path=dataset_path, 
        device=device,
        context_frames=context_frames,
        num_actions=num_actions,
        denoising_steps=denoising_steps,
        pixel_space=pixel_space,
        input_size=input_size,
        in_channels=in_channels,
        patch_size=patch_size,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads
    )

    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    video_folder = "dream_videos"
    os.makedirs(video_folder, exist_ok=True)
    env = VecVideoRecorder(
        env,
        video_folder=video_folder,
        record_video_trigger=lambda x: x == 0,
        video_length=steps,
        name_prefix=f"dream_loop"
    )

    if os.path.exists(agent_path):
        print(f"- Loading existing agent from {agent_path}")
        model = PPO.load(agent_path, env=env, device=device)
    else:
        print("- Initializing new PPO agent")
        model = PPO("CnnPolicy", env, verbose=1, learning_rate=3e-4, device=device)

    print(f"- Training agent in Dream for {steps} steps...")
    model.learn(total_timesteps=steps)
    model.save(agent_path)
    print(f"- Agent saved to {agent_path}")
    
    env.close()