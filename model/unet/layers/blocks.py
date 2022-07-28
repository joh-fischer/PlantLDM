import torch
import torch.nn as nn
from model.unet.layers.attention import Attention
from model.unet.layers.residual import ResidualBlock


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 n_groups: int = 8, attention: nn.Module = None):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_emb_dim, n_groups=n_groups)
        self.attn = attention if attention is not None else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)

        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 n_groups: int = 8, attention: nn.Module = None):
        super().__init__()
        # in_channels + out_channels as we concatenate the output of the same resolution
        # of the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_emb_dim, n_groups=n_groups)
        self.attn = attention if attention is not None else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)

        return x


class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, time_emb_dim: int, n_groups: int = 8, n_heads: int = 4):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_emb_dim, n_groups=n_groups)
        self.attn = Attention(n_channels, n_heads=n_heads)
        self.res2 = ResidualBlock(n_channels, n_channels, time_emb_dim, n_groups=n_groups)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)

        return x


class DownSample(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)

        return x


class UpSample(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)

        return x


if __name__ == "__main__":
    from model.unet.layers.attention import LinearAttention
    bs = 16
    t_emb_dim = 4

    time = torch.randn((bs, t_emb_dim))
    ipt = skip = torch.randn((bs, 8, 128, 128))
    print("Input:", ipt.shape)

    # simulate U-Net
    print("Down Block")
    db = DownBlock(8, 16, t_emb_dim, attention=LinearAttention(16), n_groups=4)
    ipt = db(ipt, time)
    print("\tRes Block:", ipt.shape)
    ds = DownSample(16)
    ipt = ds(ipt)
    print("\tDown sample:", ipt.shape)

    mb = MiddleBlock(16, t_emb_dim)
    ipt = mb(ipt, time)
    print("Middle Block:", ipt.shape)

    print("Up Block")
    us = UpSample(16)
    ipt = us(ipt)
    print("\tUp sample:", ipt.shape)
    ipt = torch.cat((ipt, skip), dim=1)
    print("\tConcat:", ipt.shape)
    ub = UpBlock(16, 8, t_emb_dim, attention=LinearAttention(8))
    ipt = ub(ipt, time)
    print("\tRes Block:", ipt.shape)
