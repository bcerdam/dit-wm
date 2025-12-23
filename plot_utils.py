import matplotlib.pyplot as plt

def plot_atari_frame(obs):
    plt.imsave('test.jpeg', obs, dpi=500)