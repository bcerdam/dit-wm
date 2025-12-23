import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def plot_atari_frame(obs):
    print(type(obs))
    img = Image.fromarray(obs)
    # img = img.resize(size=(64, 64))
    img.save('test.jpeg')

    img_array = np.array(img)
    print(f'shape: {img_array.shape}')
    # obs = torch.tensor(obs).view(3, 210, 160)
    # resized = transforms.Resize((64, 64))
    # resized_image = resized(obs)
    # print(resized_image.view(64, 64, 3).shape)
    # plt.imsave('test.jpeg', resized_image.view(64, 64, 3), dpi=500)