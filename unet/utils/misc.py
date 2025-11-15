import yaml
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import torch

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return edict(config)

def to_np(x, swap_axis=True):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if swap_axis and x.ndim == 3:
            x = x.permute(1, 2, 0)
        x = x.numpy()
    return x

def plot_images_to_file(images, filename="tmp.png"):
    n = len(images)
    plt.figure(figsize=(4 * n, 4))
    for i, img in enumerate(images):
        plt.subplot(1, n, i + 1) 
        
        img = to_np(img)
        if img.ndim == 3 and img.shape[2] == 3:
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            plt.imshow(img)
        elif img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            plt.imshow(img.squeeze(), cmap="tab20c")
        else:
            plt.imshow(img)
        plt.axis("off")
        
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()