from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import random
import torchvision.transforms.functional as TF
import torch
import numpy as np


class VOCSegmentationDataset(Dataset):
    def __init__(self, cfg, mode="train"):
        self.cfg = cfg
        self.mode = mode
        self.image_size = cfg.DATA.image_size

        norm_mean = cfg.DATA.TRANSFORMS.normalize_mean
        norm_std = cfg.DATA.TRANSFORMS.normalize_std

        if mode == "train":
            self.image_dir = cfg.DATA.trainval_image_dir
            self.mask_dir = cfg.DATA.trainval_mask_dir
            split_file = cfg.DATA.train_set_file
            self.flip_prob = cfg.DATA.TRANSFORMS.train_random_flip_prob

        elif mode == "val":
            self.image_dir = cfg.DATA.trainval_image_dir
            self.mask_dir = cfg.DATA.trainval_mask_dir
            split_file = cfg.DATA.val_set_file
            self.flip_prob = 0.0  # we don't want flipping during validation

        elif mode == "test":
            self.image_dir = cfg.DATA.test_image_dir
            self.mask_dir = cfg.DATA.test_mask_dir
            split_file = cfg.DATA.test_set_file
            self.flip_prob = 0.0

        else:
            raise ValueError(f"Error: '{mode} is an invalid mode'")

        with open(split_file, "r") as f:
            self.image_files = [line.strip() for line in f.readlines() if line.strip()]

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std),
            ]
        )

        self.mask_transform = transforms.Resize(
            (self.image_size, self.image_size), interpolation=transforms.InterpolationMode.NEAREST_EXACT
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name + ".jpg")
        image = Image.open(img_path).convert("RGB")

        if self.mode == "test":
            image = self.image_transform(image)
            return image, img_name

        mask_path = os.path.join(self.mask_dir, img_name + ".png")
        mask = Image.open(mask_path)

        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        mask = torch.from_numpy(np.array(mask)).long()
        mask[mask == 255] = 0

        return image, mask
