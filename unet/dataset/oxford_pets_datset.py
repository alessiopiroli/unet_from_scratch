import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random

class OxfordPetsDataset(Dataset):
    def __init__(self, cfg, split="trainval"):
        self.cfg = cfg
        self.split = split
        self.image_size = cfg.DATA.image_size

        root_dir = cfg.DATA.root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "annotations", "trimaps")

        if split == "trainval":
            list_path = os.path.join(root_dir, cfg.DATA.train_split_file)
            self.flip_prob = cfg.DATA.TRANSFORMS.train_random_flip_prob
        elif split == "test":
            list_path = os.path.join(root_dir, cfg.DATA.test_file)
            self.flip_prob = 0.0

        with open(list_path, "r") as f:
            self.ids = [line.split()[0] for line in f]

        norm_mean = cfg.DATA.TRANSFORMS.normalize_mean
        norm_std = cfg.DATA.TRANSFORMS.normalize_std

        self.img_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std),
            ]
        )

        self.mask_transform = transforms.Compose(
            [transforms.Resize((self.image_size, self.image_size), interpolation=Image.NEAREST)]
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]

        img_path = os.path.join(self.image_dir, name + ".jpg")
        mask_path = os.path.join(self.mask_dir, name + ".png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if random.random() < self.flip_prob:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        image = self.img_transform(image)
        mask = self.mask_transform(mask)

        mask = torch.from_numpy(np.array(mask)).long()

        mask[mask == 2] = 0
        mask[mask == 3] = 0

        return image, mask
