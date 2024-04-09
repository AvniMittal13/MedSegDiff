import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nca_diff import DiffusionNCA_Multi
from .nn import layer_norm

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

class FFParser(nn.Module):
    def __init__(self, dim=4, img_size = 32):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, img_size, img_size//2+1, 2, dtype=torch.float32) * 0.02)
        # self.w = w
        # self.h = h

    def forward(self, x, spatial_size=None):
        B, C, H, W = x.shape
        assert H == W, "height and width are not equal"
        if spatial_size is None:
            a = b = H
        else:
            a, b = spatial_size

        # x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')

        x = x.reshape(B, C, H, W)

        return x


class MedSegDiff_NCA(nn.Module):
    r"""Implementation of Diffusion NCA
    """
    def __init__(self, channel_n = 64, fire_rate=0.5, device="cuda:0", hidden_size=256, input_channels=1, drop_out_rate=0.25, img_size=32):
        r"""Init function
        """
        super(MedSegDiff_NCA, self).__init__()

        ## nca models in levels
        ## nca1 - downsampled image, nca2 - patches
        self.channel_n = channel_n
        self.device = device

        self.nca_noise = DiffusionNCA_Multi(channel_n-4, fire_rate, device, hidden_size, input_channels, drop_out_rate)
        self.nca_img = DiffusionNCA_Multi(channel_n-2, fire_rate, device, hidden_size, input_channels, drop_out_rate)
        self.nca_final = DiffusionNCA_Multi(channel_n, fire_rate, device, hidden_size, input_channels, drop_out_rate)

        self.ffparser = FFParser(dim = channel_n-4, img_size = img_size)
        # img_size and img_size//4

    def enhance(self, c, h):
        cu = layer_norm(c.shape[1:]).to(self.device)(c)
        hu = layer_norm(h.shape[1:]).to(self.device)(h)
        return cu * hu * h

    def forward(self, x, t=0):  

        num_chans = self.channel_n-1-x.shape[1]      

        x1 = self.seed(x[:, -1:, :, :], num_chans+1, 0)
        y1 = self.nca_noise(x1, t)
        print(y1.shape)
        y1 = self.ffparser(y1)  # batch, channel, height, width
        # y1 = y1.view(x.shape[0], self.channel_n, -1)

        # pass to ff parser

        x2 = self.seed(x[:, :-1, :, :], num_chans+x.shape[1]-1, 0)
        y2 = self.nca_img(x2, t)

        print(y2[:,x.shape[1]-1:,:,:].shape, y1[:,1:,:,:].shape)
        combined = self.enhance(y1[:,1:,:,:], y2[:,x.shape[1]-1:,:,:])
        # combined = y + y1[:,1:,:,:]
        x3 = torch.cat((y2[:, :x.shape[1]-1, :, :], y1[:, :1, :, :], combined), dim=1)
        x3 = self.seed(x3, self.channel_n, 1)
        y3 = self.nca_final(x3, t)
    
        # return y2
        return y3[:, :1, :, :], y3[:, 4, :, :]
        # return y3
        
    def seed(self, x, channels, start=1):
        seed = torch.zeros((x.shape[0], channels , x.shape[2], x.shape[3],), dtype=torch.float32, device=self.device)
        in_channels = x.shape[1]
        seed[:, start:start+in_channels, :,:] = x
        return seed