import yaml
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import torch
import time
import os
import logging
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

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

def setup_logging(config):
    base_dir = config.LOGGING.logging_dir
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_dir = os.path.join(base_dir, f'exp_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)
    
    log_file = os.path.join(experiment_dir, "log.log")
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode="a")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    tb_dir = os.path.join(experiment_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    
    logger.info(f"Experiment logs saved to: {experiment_dir}")
    
    return logger, writer, experiment_dir