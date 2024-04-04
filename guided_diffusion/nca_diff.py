import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from einops.layers.torch import Rearrange
from scipy import signal
import numpy as np
import torch.utils.checkpoint as checkpoint

 
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
    
# class DiffusionNCA(BackboneNCA):
#     r"""Implementation of Diffusion NCA
#     """
#     def __init__(self, channel_n = 64, fire_rate=0.5, device="cuda:0", hidden_size=256, input_channels=1, drop_out_rate=0.25, img_size=32):
#         r"""Init function
#         """
#         super(DiffusionNCA, self).__init__(channel_n, fire_rate, device, hidden_size)
#         self.drop0 = nn.Dropout(drop_out_rate)
#         self.bn0 = nn.BatchNorm2d(hidden_size) 
#         # self.norm0 = nn.LayerNorm([img_size, img_size, hidden_size])

#         # Complex
#         self.complex = False
#         if self.complex:
#             self.p0 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect", dtype=torch.complex64)
#             self.p1 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect", dtype=torch.complex64)
#             self.fc0 = nn.Linear(channel_n*3, hidden_size, dtype=torch.complex64)
#             self.fc1 = nn.Linear(hidden_size, channel_n, bias=False, dtype=torch.complex64)
#         #self.p0 = torch.conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
#         #self.p1 = torch.conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")

#     def update(self, x_in, fire_rate):
#         r"""
#         stochastic update stage of NCA
#         :param x_in: perception vector
#         :param fire_rate:
#         :param angle: rotation
#         :return: residual updated vector
#         """
        
#         x = x_in.transpose(1, 3)

#         dx = self.perceive(x)
#         dx = dx.transpose(1, 3)

#         dx = self.fc0(dx)
#         dx = F.leaky_relu(dx)  # .relu(dx)

#         dx = dx.transpose(1, 3)
#         dx = self.bn0(dx)
#         dx = dx.transpose(1, 3)

#         # dx = self.norm0(dx)
#         dx = self.drop0(dx)

#         dx = self.fc1(dx)

#         if fire_rate is None:
#             fire_rate = self.fire_rate
#         stochastic = torch.rand([dx.size(0), dx.size(1), dx.size(2), 1]) > fire_rate
#         stochastic = stochastic.float().to(self.device)
#         dx = dx * stochastic

#         x = x + dx.transpose(1, 3)
#         x = x.transpose(1, 3)

#         return x
    
#     def forward(self, x, t=0, steps=10, fire_rate=0.5):
#         r"""
#         forward pass from NCA
#         :param x: perception
#         :param steps: number of steps, such that far pixel can communicate
#         :param fire_rate:
#         :param angle: rotation
#         :return: updated input
#         """
#         # print("time", t)
        
#         # print("x shape ", x.shape)
#         x_min, x_max = torch.min(x), torch.max(x)
#         abs_max = torch.max(torch.abs(x_min), torch.abs(x_max))
        
#         x = self.seed(x)
#         x = x.transpose(1, 3)
#         # print("x shape2 ", x.shape)
#         t = torch.tensor(t).to(self.device)

#         scaled_t = t.expand(1, x.shape[1], x.shape[2], x.shape[0])#,x.shape[2],1
#         scaled_t = scaled_t.transpose(0, 3)
#         x[:, :, :, -1:] = scaled_t

#         # Add pos
#         if True:
#             x_count = torch.linspace(0, 1, x.shape[1]).expand(x.shape[0], 1, x.shape[2], x.shape[1]).transpose(1,3)
#             x_count = (x_count + torch.transpose(x_count, 1,2)) / 2
#             x[:, :, :, -2:-1] = x_count
        
#         for step in range(steps):
#             x_update = self.update(x, fire_rate).clone()
#             x = x_update
#         x = x.transpose(1,3)
#         return x[:, :2, :, :], x[:, 3, :, :]
#         # return x[:, :1, :, :], x[:, 1, :, :]
    
#     def seed(self, x):
#         seed = torch.zeros((x.shape[0], self.channel_n, x.shape[2], x.shape[3],), dtype=torch.float32, device=self.device)
#         in_channels = x.shape[1]
#         seed[:, 3:3+in_channels, :,:] = x
#         return seed

class DiffusionNCA(BackboneNCA):
    def __init__(self, channel_n = 64, fire_rate=0.5, device="cuda:0", hidden_size=256, input_channels=1, drop_out_rate=0.25, img_size=32):
        super(DiffusionNCA, self).__init__(channel_n, fire_rate, device, hidden_size)
        self.drop0 = nn.Dropout(drop_out_rate)
        self.bn0 = nn.BatchNorm2d(hidden_size) 

    def update(self, x_in, fire_rate):
        x = x_in.transpose(1, 3)
        dx = self.perceive(x)
        dx = dx.transpose(1, 3)
        dx = self.fc0(dx)
        dx = F.leaky_relu(dx)
        dx = dx.transpose(1, 3)
        dx = self.bn0(dx)
        dx = dx.transpose(1, 3)
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
        x_min, x_max = torch.min(x), torch.max(x)
        abs_max = torch.max(torch.abs(x_min), torch.abs(x_max))
        
        x = self.seed(x)
        x = x.transpose(1, 3)
        t = torch.tensor(t).to(self.device)
        scaled_t = t.expand(1, x.shape[1], x.shape[2], x.shape[0])
        scaled_t = scaled_t.transpose(0, 3)
        x[:, :, :, -1:] = scaled_t

        if True:
            x_count = torch.linspace(0, 1, x.shape[1]).expand(x.shape[0], 1, x.shape[2], x.shape[1]).transpose(1,3)
            x_count = (x_count + torch.transpose(x_count, 1,2)) / 2
            x[:, :, :, -2:-1] = x_count
        
        # Define a sequence of layers to checkpoint
        layers_to_checkpoint = [
            lambda x: self.update(x, fire_rate).clone() for _ in range(steps)
        ]
        
        # Checkpoint the sequence of layers
        x = checkpoint.checkpoint_sequential(layers_to_checkpoint, steps, x)
        
        x = x.transpose(1,3)
        return x[:, :1, :, :], x[:, 1, :, :]
    
    def seed(self, x):
        seed = torch.zeros((x.shape[0], self.channel_n, x.shape[2], x.shape[3],), dtype=torch.float32, device=self.device)
        in_channels = x.shape[1]
        seed[:, 1:1+in_channels, :,:] = x
        return seed


class DiffusionNCA_Multi(BackboneNCA):
    r"""Implementation of Diffusion NCA
    """
    def __init__(self, channel_n = 64, fire_rate=0.5, device="cuda:0", hidden_size=256, input_channels=1, drop_out_rate=0.25, img_size=32):
        r"""Init function
        """
        super(DiffusionNCA_Multi, self).__init__(channel_n, fire_rate, device, hidden_size)
        self.drop0 = nn.Dropout(drop_out_rate)
        self.bn0 = nn.BatchNorm2d(hidden_size) 
        # self.norm0 = nn.LayerNorm([img_size, img_size, hidden_size])

        # Complex
        self.complex = False
        if self.complex:
            self.p0 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect", dtype=torch.complex64)
            self.p1 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect", dtype=torch.complex64)
            self.fc0 = nn.Linear(channel_n*3, hidden_size, dtype=torch.complex64)
            self.fc1 = nn.Linear(hidden_size, channel_n, bias=False, dtype=torch.complex64)
        #self.p0 = torch.conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        #self.p1 = torch.conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")

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

        dx = dx.transpose(1, 3)
        dx = self.bn0(dx)
        dx = dx.transpose(1, 3)

        # dx = self.norm0(dx)
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

        x_min, x_max = torch.min(x), torch.max(x)
        abs_max = torch.max(torch.abs(x_min), torch.abs(x_max))
        
        # x = self.seed(x)
        x = x.transpose(1, 3)
        t = torch.tensor(t).to(self.device)

        scaled_t = t.expand(1, x.shape[1], x.shape[2], x.shape[0])#,x.shape[2],1
        scaled_t = scaled_t.transpose(0, 3)
        x[:, :, :, -1:] = scaled_t

        # Add pos
        if True:
            x_count = torch.linspace(0, 1, x.shape[1]).expand(x.shape[0], 1, x.shape[2], x.shape[1]).transpose(1,3)
            x_count = (x_count + torch.transpose(x_count, 1,2)) / 2
            x[:, :, :, -2:-1] = x_count
        
        # for step in range(steps):
        #     x_update = self.update(x, fire_rate).clone()
        #     x = x_update
        
        layers_to_checkpoint = [
            lambda x: self.update(x, fire_rate).clone() for _ in range(steps)
        ]
        
        # Checkpoint the sequence of layers
        x = checkpoint.checkpoint_sequential(layers_to_checkpoint, steps, x)
        x = x.transpose(1,3)
        # print(x.shape)
        return x

    ############################


    
class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, out_dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(torch.nn.Module):
    def __init__(self, dim, heads=8, head_dim=64, dropout=0.):
        super().__init__()
        inner_dim = head_dim * heads
        project_out = not (heads == 1 and head_dim == dim)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.to_qkv = torch.nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = torch.nn.Softmax(dim=-1)

        self.mask_heads = None
        self.attn_map = None

        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, dim),
            torch.nn.Dropout(dropout)
        ) if project_out else torch.nn.Identity()

    def forward(self, x, localize=None, h=None, w=None, **kwargs):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        if localize is not None:
            q = rearrange(q, 'b h n d -> b h n 1 d')
            k = localize(k, h, w)  # b h n (attn_height attn_width) d
            v = localize(v, h, w)  # b h n (attn_height attn_width) d

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # b h n 1 (attn_height attn_width)

        attn = self.attend(dots)  # b h n 1 (attn_height attn_width)

        if kwargs.get('mask', False):
            mask = kwargs['mask']
            assert len(mask) <= attn.shape[1], 'number of heads to mask must be <= number of heads'
            attn[:, mask] *= 0.0

        self.attn_maps = attn

        out = torch.matmul(attn, v)  # b h n 1 d
        out = rearrange(out, 'b h n 1 d -> b n (h d)') if localize else rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(torch.nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(torch.nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, head_dim=head_dim, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dim, dropout=dropout))
            ]))

    def encode(self, x, attn, ff, localize_attn_fn=None, h=None, w=None, **kwargs):
        x = attn(x, localize=localize_attn_fn, h=h, w=w, **kwargs) + x
        x = ff(x) + x
        return x

    def forward(self, x, localize_attn_fn=None, h=None, w=None, **kwargs):
        if self.training and len(self.layers) > 1:
            # gradient checkpointing to save memory but at the cost of re-computing forward pass during backward pass
            funcs = [lambda _x: self.encode(_x, attn, ff, localize_attn_fn, h, w, **kwargs) for attn, ff in self.layers]
            x = torch.utils.checkpoint.checkpoint_sequential(funcs, segments=len(funcs), input=x)
        else:
            for attn, ff in self.layers:
                x = self.encode(x, attn, ff, localize_attn_fn, h, w, **kwargs)
        return x


class LocalizeAttention(torch.nn.Module):
    def __init__(self, attn_neighbourhood_size, device) -> None:
        super().__init__()
        self.attn_neighbourhood_size = attn_neighbourhood_size
        self.device = device
        self.attn_filters = neighbourhood_filters(self.attn_neighbourhood_size, self.device)

    def forward(self, x, height, width):
        '''attn_filters: [filter_n, h, w]'''
        b, h, _, d = x.shape
        y = rearrange(x, 'b h (i j) d -> (b h d) 1 i j', i=height, j=width)
        y = torch.nn.functional.conv2d(y, self.attn_filters[:, None], padding='same')
        _x = rearrange(y, '(b h d) filter_n i j -> b h (i j) filter_n d', b=b, h=h, d=d)
        return _x

def neighbourhood_filters(neighbourhood_size, device):
    height, width = neighbourhood_size
    impulses = []
    for i in range(height):
        for j in range(width):
            impulse = signal.unit_impulse((height, width), idx=(i,j), dtype=np.float32)
            impulses.append(impulse)
    filters = torch.tensor(np.stack(impulses), device=device)
    return filters


class Diffusion_ViTCA_NCA(BackboneNCA):
    r"""Implementation of Diffusion NCA
    """
    def __init__(self, channel_n = 64, fire_rate=0.5, device="cuda", hidden_size=128, input_channels=1, drop_out_rate=0.25, img_size=28,
                 depth=1, heads=4, mlp_dim=64, dropout=0., embed_dim = 16):
        r"""Init function
        """
        super(Diffusion_ViTCA_NCA, self).__init__(channel_n, fire_rate, device, hidden_size)
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


        ## vitca params:

        self.localized_attn_neighbourhood = [3,3]
        self.localize_attn_fn = LocalizeAttention(self.localized_attn_neighbourhood, device)
        
        embed_dim = channel_n

        self.transformer = Transformer(embed_dim, depth, heads, embed_dim // heads, mlp_dim, dropout)
        self.mlp_head = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, self.channel_n)
        )

        # don't update cells before first backward pass or else cell grid will have immensely diverged and grads will
        # be large and unhelpful
        self.mlp_head[1].weight.data.zero_()
        self.mlp_head[1].bias.data.zero_()

    def vit_positional_encoding(self, n, dim, device='cpu'):
        position = torch.arange(n, device=device).unsqueeze(1)
        div_term_even = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
        div_term_odd = torch.exp(torch.arange(1, dim, 2, device=device) * (-math.log(10000.0) / dim))
        pe = torch.zeros(n, 1, dim, device=device)
        pe[:, 0, 0::2] = torch.sin(position * div_term_even)
        pe[:, 0, 1::2] = torch.cos(position * div_term_odd)
        return pe.transpose(0, 1)


    def perceive(self, x):
        r"""Perceptive function, combines learnt conv outputs with the identity of the cell
            #Args:
                x: image
        """
        batch_size, height, width, n_channels = x.size() 

        y = rearrange(x, 'b h w c -> b (h w) c')

        pe = self.vit_positional_encoding(y.shape[-2], y.shape[-1], self.device)
        y = y+pe

        cells = rearrange(x, 'b h w c -> b c h w')
        y = self.transformer(y, localize_attn_fn=self.localize_attn_fn, h=cells.shape[-2], w=cells.shape[-1])

        return y   

    
    def update(self, x_in, fire_rate):
        r"""
        stochastic update stage of NCA
        :param x_in: perception vector
        :param fire_rate:
        :param angle: rotation
        :return: residual updated vector
        """

        x = x_in
        dx = self.perceive(x_in)
        dx = self.mlp_head(dx)
        dx = rearrange(dx, 'b (h w) c -> b h w c', h=x_in.shape[1], w = x_in.shape[2])

        if fire_rate is None:
            fire_rate = self.fire_rate
        stochastic = torch.rand([dx.size(0), dx.size(1), dx.size(2), 1]) > fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic
        
        x = x + dx

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
        #print(t)
        
        #print("x shape ", x.shape)
        x_min, x_max = torch.min(x), torch.max(x)
        abs_max = torch.max(torch.abs(x_min), torch.abs(x_max))
        
        x = self.seed(x)
        x = x.transpose(1, 3)
        #print("x shape2 ", x.shape)
        t = torch.tensor(t).to(self.device)

        scaled_t = t.expand(1, x.shape[1], x.shape[2], x.shape[0])#,x.shape[2],1
        scaled_t = scaled_t.transpose(0, 3)
        x[:, :, :, -1:] = scaled_t

        # Add pos
        if True:
            x_count = torch.linspace(0, 1, x.shape[1]).expand(x.shape[0], 1, x.shape[2], x.shape[1]).transpose(1,3)
            x_count = (x_count + torch.transpose(x_count, 1,2)) / 2
            x[:, :, :, -2:-1] = x_count
        

        # Define a sequence of layers to checkpoint
        layers_to_checkpoint = [
            lambda x: self.update(x, fire_rate).clone() for _ in range(steps)
        ]
        
        # Checkpoint the sequence of layers
        x = checkpoint.checkpoint_sequential(layers_to_checkpoint, steps, x)
        
        # for step in range(steps):
        #     x_update = self.update(x, fire_rate).clone()
        #     x = x_update
        x = x.transpose(1,3)

        return x[:, :1, :, :], x[:, 1, :, :]

    def seed(self, x):
        seed = torch.zeros((x.shape[0], self.channel_n, x.shape[2], x.shape[3],), dtype=torch.float32, device=self.device)
        in_channels = x.shape[1]
        seed[:, 3:3+in_channels, :,:] = x
        return seed


############## cbam attention
class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        output = self.conv(concat)
        output = F.sigmoid(output) * x 
        return output 

class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear(max.view(b,c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg.view(b,c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = F.sigmoid(output) * x
        return output

class CBAMDiffNCA(DiffusionNCA):
    r"""
        NCA to include global image information
    """
    def __init__(self, channel_n = 64, fire_rate=0.5, device="cuda", hidden_size=256, input_channels=1, output_channel = 1, drop_out_rate=0.5, img_size=28, steps = 10, bias = False, r = 16):
        # self, channel_n = 64, fire_rate=0.5, device="cuda:0", hidden_size=256, input_channels=1, drop_out_rate=0.25, img_size=32
        super(CBAMDiffNCA, self).__init__(channel_n, fire_rate, device, hidden_size, input_channels, drop_out_rate, img_size)
        self.sam = SAM(bias)
        self.cam = CAM(channels=channel_n, r=r)

    def cbam_attention(self, x):
        out = self.cam(x)
        out = self.sam(out)
        
        x = x+out
        print(x.shape)
        return x 

    def perceive(self, x):
        r"""Perceptive function, combines 2 conv outputs with the identity of the cell
            #Args:
                x: image
        """
        x = self.cbam_attention(x)

        y1 = self.p0(x)
        y2 = self.p1(x)
        # y3 = self.p2(x)

        y = torch.cat((x,y1,y2),1)

        return y
    