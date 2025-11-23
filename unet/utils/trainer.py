from unet.utils.misc import setup_logging
from unet.model.u_net_model import UNet
import torch
from unet.dataset.oxford_pets_datset import OxfordPetsDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import os


class Trainer:
    def __init__(self, config):
        self.config = config
        self.num_epochs = self.config.TRAIN.num_epochs
        self.logger, self.writer, self.experiment_dir = setup_logging(self.config)
        self.build_model()
        self.build_dataloaders()

    def build_model(self):
        self.model = UNet(self.config).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.TRAIN.lr)
        self.logger.info("Built model and optimizer")

    def build_dataloaders(self):
        train_dataset = OxfordPetsDataset(self.config, split="trainval")
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.DATA.batch_size,
            shuffle=True,
            num_workers=self.config.DATA.num_workers,
        )

        test_dataset = OxfordPetsDataset(self.config, split="test")
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.config.DATA.batch_size,
            shuffle=False,
            num_workers=self.config.DATA.num_workers,
        )

        self.logger.info("Built Dataloader")

    def compute_loss(self, output, labels):
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fn(output, labels)
        mask_bg = labels == 0
        mask_fg = labels != 0

        loss_bg = loss[mask_bg].mean() if mask_bg.sum() > 0 else 0.0
        loss_fg = loss[mask_fg].mean() if mask_fg.sum() > 0 else 0.0

        total_loss = loss_bg + loss_fg
        return total_loss

    def post_processor(self, output):
        output_sm = torch.nn.functional.softmax(output, 1)
        output_idxs = torch.argmax(output_sm, 1)
        return output_idxs

    def train_one_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        for img, labels in self.train_dataloader:
            img, labels = img.cuda(), labels.cuda()

            self.optimizer.zero_grad()
            output = self.model(img)
            loss = self.compute_loss(output, labels)

            epoch_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        return epoch_loss / len(self.train_dataloader)

    def evaluate(self):
        self.model.eval()
        total_intersect_fg = 0
        total_union_fg = 0
        total_intersect_bg = 0
        total_union_bg = 0
        validation_loss = 0.0

        with torch.no_grad():
            for img, labels in self.test_dataloader:
                img, labels = img.cuda(), labels.cuda()
                output = self.model(img)
                validation_loss += self.compute_loss(output, labels).item()

                predictions = self.post_processor(output)
                total_intersect_fg += ((predictions == 1) & (labels == 1)).sum().item()
                total_union_fg += ((predictions == 1) | (labels == 1)).sum().item()
                total_intersect_bg += ((predictions == 0) & (labels == 0)).sum().item()
                total_union_bg += ((predictions == 0) | (labels == 0)).sum().item()

            iou_fg = total_intersect_fg / (total_union_fg + 1e-6)
            iou_bg = total_intersect_bg / (total_union_bg + 1e-6)
            miou = (iou_fg + iou_bg) / 2.0
            val_loss = validation_loss / len(self.test_dataloader)

            return miou, val_loss

    def train(self):
        self.logger.info("Starting training...")
        for epoch in range(self.num_epochs):
            train_loss = self.train_one_epoch()
            miou, val_loss = self.evaluate()

            self.logger.info(
                f"Epoch: {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, mIoU: {miou:.4f}"
            )
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Metric/mIoU", miou, epoch)

            save_path = os.path.join(self.experiment_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(self.model.state_dict(), save_path)

        self.logger.info("Training finished.")

    def load_model(self, ckpt):
        self.model.load_state_dict(torch.load(ckpt, weights_only=True))
        self.model.eval()

    def evaluate_model(self, ckpt):
        self.logger.info("Started model evaluation...")
        self.load_model(ckpt)
        miou, val_loss = self.evaluate()
        self.logger.info(f"mIoU: {miou:.4f}, Loss: {val_loss:.4f}")
