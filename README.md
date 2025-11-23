# U-Net from scratch

<div style="display: flex; gap: 5px;">
    <a href="https://arxiv.org/abs/1505.04597">
        <img src="https://img.shields.io/badge/Arxiv-Paper-green" alt="Arxiv Paper" />
    </a>
</div>

###
Implementing ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597) from scratch.

### Clone and install dependencies
``` 
git clone https://github.com/alessiopiroli/u_net_from_scratch.git
pip install -r requirements.txt 
```

### Train 
``` 
python train.py unet/config/unet_config.yaml
```

### Evaluate 
``` 
python evaluate.py unet/config/unet_config.yaml path/to/ckpt.pt
```

### Qualitative Results
> Qualitative semantic segmentation results on the Oxford-IIIT Pet Dataset validation split, with the UNet model trained from scratch for 30 epochs.
>
> ![](assets/predict.gif)
