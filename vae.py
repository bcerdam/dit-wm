import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


'''
encode_observation(): Recieves pixel space observation and encodes it to latent space.

observation: Atari enviroment pixel space observation
vae: Off the shelf VAE
data_collection: Bool that indicates wether we are collecting data or not
'''
def encode_observation(observation, vae, data_collection=True, device='cuda'):
    
    with torch.no_grad():
        pil_image_resized = Image.fromarray(observation).resize(size=(64, 64))
        observation_tensor = torch.tensor(np.array(pil_image_resized)).float().permute(2, 0, 1).unsqueeze(0).to(device)
        observation_tensor_normalized = (observation_tensor / 127.5) - 1.0
        observation_latent = vae.encode(observation_tensor_normalized).latent_dist.sample() * 0.18215

        if data_collection == True:
            return observation_latent.cpu().numpy()
        else:
            return observation_latent
    

'''
decode_latent(): Recieves latent space observation, and decodes it into pixel space.

latent: Encoded observation from Atari enviroment
vae: Off the shelf VAE
'''
def decode_latent(latent, vae, device='cuda'):
    latent = torch.tensor(latent).to(device)
    with torch.no_grad():
        decoded_latent = vae.decode(latent/ 0.18215).sample
        decoded_latent = (decoded_latent / 2 + 0.5).clamp(0, 1)
        img_decoded_latent = decoded_latent.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (img_decoded_latent*255).astype(np.uint8)      





