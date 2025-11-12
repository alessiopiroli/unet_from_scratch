import torch
from torch.utils.data import DataLoader
from unet.dataset.voc_dataset import VOCSegmentationDataset
from unet.utils.misc import load_config
from pathlib import Path
import pytest
import os

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "unet" / "config" / "unet_config.yaml"
cfg = load_config(CONFIG_PATH)

train_split_file = PROJECT_ROOT / cfg.DATA.train_set_file
DATA_IS_PRESENT = os.path.exists(train_split_file)
SKIP_REASON = f"Data file not found at {train_split_file}, skipping data-dependent tests"

@pytest.mark.skipif(not DATA_IS_PRESENT, reason=SKIP_REASON)
def test_dataset_init_len():
    train_dataset = VOCSegmentationDataset(cfg, mode="train")
    assert len(train_dataset) > 0, "Train dataset has 0 items"

    val_dataset = VOCSegmentationDataset(cfg, mode="val")
    assert len(val_dataset) > 0, "Validation dataset has 0 items"

    test_dataset = VOCSegmentationDataset(cfg, mode="test")
    assert len(test_dataset) > 0, "Test dataset has 0 items"

@pytest.mark.skipif(not DATA_IS_PRESENT, reason=SKIP_REASON)
def test_dataset_getitem_train():
    dataset = VOCSegmentationDataset(cfg, mode="train")
    image, mask = dataset[0]

    img_size = cfg.DATA.image_size
    n_classes = cfg.DATA.n_classes

    assert isinstance(image, torch.Tensor), "Image is not a Tensor"
    assert image.shape == (3, img_size, img_size), f"Image shape is {image.shape}"
    assert image.dtype == torch.float32, "Image is not float32"

    assert isinstance(mask, torch.Tensor), "Mask is not a Tensor"
    assert mask.shape == (img_size, img_size), f"Mask shape is {mask.shape}"
    assert mask.dtype == torch.long, "Mask is not long"

    assert mask.min() >= 0, "Mask contains values < 0"
    assert mask.max() < n_classes, "Mask contains values >= n_classes (21)"
    assert torch.all(mask != 255), "Mask still contains 255"

@pytest.mark.skipif(not DATA_IS_PRESENT, reason=SKIP_REASON)
def test_dataset_getitem_test():
    dataset = VOCSegmentationDataset(cfg, mode="test")
    item = dataset[0]
    
    assert len(item) == 2, "Test item does not have 2 elements"
    image, img_name = item
    
    assert isinstance(image, torch.Tensor), "Test image is not a Tensor"
    assert isinstance(img_name, str), "Image name is not a string"
    assert image.shape == (3, cfg.DATA.image_size, cfg.DATA.image_size)

@pytest.mark.skipif(not DATA_IS_PRESENT, reason=SKIP_REASON)
def test_dataloader_batch():
    dataset = VOCSegmentationDataset(cfg, mode="train")
    loader = DataLoader(
        dataset, 
        batch_size=4, 
        num_workers=cfg.DATA.num_workers,
        shuffle=True
    )
    
    try:
        image_batch, mask_batch = next(iter(loader))
    except Exception as e:
        assert False, f"DataLoader failed to get a batch: {e}"
    
    assert image_batch.shape == (4, 3, cfg.DATA.image_size, cfg.DATA.image_size)
    assert mask_batch.shape == (4, cfg.DATA.image_size, cfg.DATA.image_size)