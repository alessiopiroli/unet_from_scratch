from unet.model.u_net_model import UNet
import argparse
from unet.utils.misc import load_config
import torch

# from unet.dataset.voc_dataset import VOCSegmentationDataset
from unet.dataset.oxford_pets_datset import OxfordPetsDataset
from torch.utils.data import DataLoader
from unet.utils.misc import plot_images_to_file
import torch.nn as nn

###########################################
import debugpy

debugpy.listen(("localhost", 6001))
print("Waiting for debugger attach...")
debugpy.wait_for_client()
###########################################


def post_processor(output):
    output_sm = torch.nn.functional.softmax(output, 1)
    output_idxs = torch.argmax(output_sm, 1)
    return output_idxs


def compute_loss(output, labels):
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fn(output, labels)
    mask_bg = labels == 0
    mask_fg = labels != 0
    loss_bg = loss[mask_bg]
    loss_fg = loss[mask_fg]
    loss_bg = torch.mean(loss_bg)
    loss_fg = torch.mean(loss_fg)
    total_loss = loss_bg + loss_fg
    return total_loss


def train_one_epoch(model, train_dataloader, optimizer):
    epoch_loss = 0.0
    for img, labels in train_dataloader:
        img, labels = img.cuda(), labels.cuda()
        optimizer.zero_grad()
        output = model(img)
        loss = compute_loss(output, labels)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    return epoch_loss / len(train_dataloader)


def train(model, train_dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model = model.cuda()
    model.train()
    n_epochs = 10
    for epoch in range(n_epochs):
        loss = train_one_epoch(model, train_dataloader, optimizer)


def main(args):
    config = load_config(args.config)
    model = UNet(config)
    # train_dataset = VOCSegmentationDataset(config, mode="train")
    train_dataset = OxfordPetsDataset("data/oxford_pets_dataset", "trainval")
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    train(model, train_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default="config/gpt_config.yaml", help="Config path")
    args = parser.parse_args()
    main(args)
