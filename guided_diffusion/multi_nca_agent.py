import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nca_diff import DiffusionNCA_Multi

class Multi_NCA(nn.Module):
    r"""Implementation of Diffusion NCA
    """
    def __init__(self, channel_n = 64, fire_rate=0.5, device="cuda:0", hidden_size=256, input_channels=1, drop_out_rate=0.25, img_size=32):
        r"""Init function
        """
        super(Multi_NCA, self).__init__()

        ## nca models in levels
        ## nca1 - downsampled image, nca2 - patches
        self.channel_n = channel_n
        self.device = device
        self.nca1 = DiffusionNCA_Multi(channel_n, fire_rate, device, hidden_size, input_channels, drop_out_rate)
        self.nca2 = DiffusionNCA_Multi(channel_n, fire_rate, device, hidden_size, input_channels, drop_out_rate)
        # img_size and img_size//4

    def forward(self, x, t=0):
        # nca2
        down_scaled_size = (int(x.shape[2] // 4), int(x.shape[3] // 4))
        x_resized = F.interpolate(x, size=down_scaled_size, mode='bilinear', align_corners=False)
        
        x1 = self.seed(x_resized)
        y1 = self.nca1(x1, t)

        # nca2 
        up = torch.nn.Upsample(scale_factor=4, mode='nearest')
        x2 = up(y1)

        x2[:, 1:1+x.shape[1], :, :] = x
        y2 = self.nca2(x2, t)
    
        # return y2
        return y2[:, :1, :, :], y2[:, 4, :, :]
        
    def seed(self, x):
        seed = torch.zeros((x.shape[0], self.channel_n, x.shape[2], x.shape[3],), dtype=torch.float32, device=self.device)
        in_channels = x.shape[1]
        seed[:, 1:1+in_channels, :,:] = x
        return seed