import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class BasicNCA(nn.Module):
    r"""Basic implementation of an NCA using a sobel x and y filter for the perception
    """
    def __init__(self, channel_n=60, fire_rate=0.5, device="cuda", hidden_size=128, input_channels=1, init_method="standard"):
        r"""Init function
            #Args:
                channel_n: number of channels per cell
                fire_rate: random activation of each cell
                device: device to run model on
                hidden_size: hidden size of model
                input_channels: number of input channels
                init_method: Weight initialisation function
        """
        super(BasicNCA, self).__init__()

        self.device = device
        self.channel_n = channel_n
        self.input_channels = input_channels

        self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        if init_method == "xavier":
            torch.nn.init.xavier_uniform(self.fc0.weight)
            torch.nn.init.xavier_uniform(self.fc1.weight)

        self.fire_rate = fire_rate
        self.to(self.device)

    def perceive(self, x):
        r"""Perceptive function, combines 2 sobel x and y outputs with the identity of the cell
            #Args:
                x: image
        """
        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.channel_n, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)

        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        dy = dx.T

        y1 = _perceive_with(x, dx)
        y2 = _perceive_with(x, dy)
        y = torch.cat((x,y1,y2),1)
        return y

    def update(self, x_in, fire_rate):
        r"""Update function runs same nca rule on each cell of an image with a random activation
            #Args:
                x_in: image
                fire_rate: random activation of cells
        """
        x = x_in.transpose(1,3)

        dx = self.perceive(x)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)

        x = x.transpose(1,3)

        return x

    def forward(self, x, steps=64, fire_rate=0.5):
        r"""Forward function applies update function s times leaving input channels unchanged
            #Args:
                x: image
                steps: number of steps to run update
                fire_rate: random activation rate of each cell
        """
        for step in range(steps):
            x2 = self.update(x, fire_rate).clone() #[...,3:][...,3:]
            x = torch.concat((x[...,:self.input_channels], x2[...,self.input_channels:]), 3)
        return x
    
class BackboneNCA(BasicNCA):
    r"""Implementation of the backbone NCA of Med-NCA
    """
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, input_channels=1):
        r"""Init function
            #Args:
                channel_n: number of channels per cell
                fire_rate: random activation of each cell
                device: device to run model on
                hidden_size: hidden size of model
                input_channels: number of input channels
        """
        super(BackboneNCA, self).__init__(channel_n, fire_rate, device, hidden_size)
        self.p0 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        self.p1 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")

    def perceive(self, x):
        r"""Perceptive function, combines 2 conv outputs with the identity of the cell
            #Args:
                x: image
        """
        y1 = self.p0(x)
        y2 = self.p1(x)
        y = torch.cat((x,y1,y2),1)
        return y
    
class DiffusionNCA(BackboneNCA):
    r"""Implementation of Diffusion NCA
    """
    def __init__(self, channel_n = 32, fire_rate=0.5, device="cuda:0", hidden_size=256, input_channels=1, drop_out_rate=0.25, img_size=32):
        r"""Init function
        """
        super(DiffusionNCA, self).__init__(channel_n, fire_rate, device, hidden_size)
        self.drop0 = nn.Dropout(drop_out_rate)
        self.norm0 = nn.LayerNorm([img_size, img_size, hidden_size])

        # Complex
        self.complex = False
        if self.complex:
            self.p0 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect", dtype=torch.complex64)
            self.p1 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect", dtype=torch.complex64)
            self.fc0 = nn.Linear(channel_n*3, hidden_size, dtype=torch.complex64)
            self.fc1 = nn.Linear(hidden_size, channel_n, bias=False, dtype=torch.complex64)
        #self.p0 = torch.conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        #self.p1 = torch.conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")

        self.ffparser = FFParser(dim = channel_n, img_size = img_size)
        self.norm_fft = nn.LayerNorm(normalized_shape=(channel_n, img_size*img_size))


    def update(self, x_in, fire_rate):
        r"""
        stochastic update stage of NCA
        :param x_in: perception vector
        :param fire_rate:
        :param angle: rotation
        :return: residual updated vector
        """
        
        x = x_in.transpose(1, 3)

        dx = self.perceive(x)
        dx = dx.transpose(1, 3)

        dx = self.fc0(dx)
        dx = F.leaky_relu(dx)  # .relu(dx)

        dx = self.norm0(dx)
        dx = self.drop0(dx)

        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate = self.fire_rate
        stochastic = torch.rand([dx.size(0), dx.size(1), dx.size(2), 1]) > fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x + dx.transpose(1, 3)
        x = x.transpose(1, 3)

        return x
    
    def forward(self, x, t=0, steps=10, fire_rate=0.5):
        r"""
        forward pass from NCA
        :param x: perception
        :param steps: number of steps, such that far pixel can communicate
        :param fire_rate:
        :param angle: rotation
        :return: updated input
        """
        # print("time", t)
        
        # print("x shape ", x.shape)
        x_min, x_max = torch.min(x), torch.max(x)
        abs_max = torch.max(torch.abs(x_min), torch.abs(x_max))

        # x = self.ffparser(x)
        
        x = self.seed(x)
        

        x = x.transpose(1, 3)
        # print("x shape2 ", x.shape)
        t = torch.tensor(t).to(self.device)

        scaled_t = t.expand(1, x.shape[1], x.shape[2], x.shape[0])#,x.shape[2],1
        scaled_t = scaled_t.transpose(0, 3)
        x[:, :, :, -1:] = scaled_t

        # Add pos
        if True:
            x_count = torch.linspace(0, 1, x.shape[1]).expand(x.shape[0], 1, x.shape[2], x.shape[1]).transpose(1,3)
            x_count = (x_count + torch.transpose(x_count, 1,2)) / 2
            x[:, :, :, -2:-1] = x_count
        
        for step in range(steps):
            
            x = x.transpose(1,3)
            y = self.ffparser(x)  # batch, channel, height, width
            y = y.view(x.shape[0], self.channel_n, -1)
            y = self.norm_fft(y)
            y = y.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
            x = y
            x = x.transpose(1, 3)

            x_update = self.update(x, fire_rate).clone()
            x = x_update
        x = x.transpose(1,3)
        # return x[:, :2, :, :], x[:, 3, :, :]
        return x[:, :1, :, :], x[:, 1, :, :]
    
    def seed(self, x):
        seed = torch.zeros((x.shape[0], self.channel_n, x.shape[2], x.shape[3],), dtype=torch.float32, device=self.device)
        in_channels = x.shape[1]
        seed[:, 1:1+in_channels, :,:] = x
        return seed