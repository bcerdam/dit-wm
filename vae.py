from diffusers.models import AutoencoderKL
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

model = "CompVis/stable-diffusion-v1-4"
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

with torch.no_grad():
    img = Image.open('test.jpeg').resize(size=(64, 64))

    img_tensor = torch.tensor(np.array(img)).float().permute(2, 0, 1).unsqueeze(0)
    img_tensor = (img_tensor / 127.5) - 1.0
    # print(img_tensor.shape)
    latent = vae.encode(img_tensor)
    # print(latent.latent_dist.sample().shape)

    decoded_latent = vae.decode(latent.latent_dist.sample())['sample']
    decoded_latent = (decoded_latent / 2 + 0.5).clamp(0, 1)
    # print(decoded_latent.shape)

    img_decoded_latent = decoded_latent.squeeze(0).permute(1, 2, 0).numpy()
    print(img_decoded_latent.shape)

    plt.imsave('test_1.jpeg', img_decoded_latent, dpi=500)