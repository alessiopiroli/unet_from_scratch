import torch
import pytest
from pathlib import Path

from unet.model.u_net_model import UNet
from unet.utils.misc import load_config

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "unet" / "config" / "unet_config.yaml"
cfg = load_config(CONFIG_PATH)

def test_model_forward_pass_shape():
    batch_size = 4
    img_size = cfg.DATA.image_size
    in_channels = cfg.MODEL.in_channels
    out_channels = cfg.MODEL.out_channels

    model = UNet(cfg)
    
    dummy_input = torch.randn(batch_size, in_channels, img_size, img_size)
    output = model(dummy_input)
    expected_shape = (batch_size, out_channels, img_size, img_size)
    assert output.shape == expected_shape