from .u_net_parts import DoubleConv, OutConv
import torch.nn as nn
import torch

class UNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.in_channels = config.MODEL.in_channels
        self.out_channels = config.MODEL.out_channels
        base_channels = config.MODEL.base_channels

        # left side of the u_net architecture
        self.left_conv_1 = DoubleConv(self.in_channels, base_channels)
        self.pool_1 = nn.MaxPool2d(kernel_size=2)

        self.left_conv_2 = DoubleConv(base_channels, base_channels*2)
        self.pool_2 = nn.MaxPool2d(kernel_size=2)

        self.left_conv_3 = DoubleConv(base_channels*2, base_channels*4)
        self.pool_3 = nn.MaxPool2d(kernel_size=2)

        self.left_conv_4 = DoubleConv(base_channels*4, base_channels*8)
        self.pool_4 = nn.MaxPool2d(kernel_size=2)

        # bottom part of the u_net architecture
        self.bottom = DoubleConv(base_channels*8, base_channels*16)

        # right side of the u_net architecture
        self.up_conv_1 = nn.ConvTranspose2d(base_channels*16, base_channels*8, kernel_size=2, stride=2)
        self.right_conv_1 = DoubleConv(base_channels*16, base_channels*8)

        self.up_conv_2 = nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=2, stride=2)
        self.right_conv_2 = DoubleConv(base_channels*8, base_channels*4)

        self.up_conv_3 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.right_conv_3 = DoubleConv(base_channels*4, base_channels*2)

        self.up_conv_4 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.right_conv_4 = DoubleConv(base_channels*2, base_channels)

        self.output_conv = OutConv(base_channels, self.out_channels)
    
    def forward(self, x):
        skip_1 = self.left_conv_1(x)
        x = self.pool_1(skip_1)
        
        skip_2 = self.left_conv_2(x)
        x = self.pool_2(skip_2)
        
        skip_3 = self.left_conv_3(x)
        x = self.pool_3(skip_3)
        
        skip_4 = self.left_conv_4(x)
        x = self.pool_4(skip_4)
        
        x = self.bottom(x)
        
        x = self.up_conv_1(x)
        x = torch.cat([skip_4, x], dim=1)
        x = self.right_conv_1(x)
        
        x = self.up_conv_2(x)
        x = torch.cat([skip_3, x], dim=1)
        x = self.right_conv_2(x)
        
        x = self.up_conv_3(x)
        x = torch.cat([skip_2, x], dim=1)
        x = self.right_conv_3(x)
        
        x = self.up_conv_4(x)
        x = torch.cat([skip_1, x], dim=1)
        x = self.right_conv_4(x)
        
        logits = self.output_conv(x)
        return logits