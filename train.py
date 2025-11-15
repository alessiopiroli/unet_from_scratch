from unet.model.u_net_model import UNet
import argparse
from unet.utils.misc import load_config
import torch
from unet.dataset.voc_dataset import VOCSegmentationDataset
from torch.utils.data import DataLoader


def main(args):
    config = load_config(args.config)
    model = UNet(config)
    train_dataset = VOCSegmentationDataset(config, mode="train")
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default="config/gpt_config.yaml", help="Config path")
    args = parser.parse_args()
    main(args)