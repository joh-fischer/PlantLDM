import torch
import torch.nn as nn
from model.unet.layers.attention import SelfAttention
from model.unet.layers.residual import ResidualBlock


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_groups: int = 32):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels, n_groups=n_groups)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)

        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_groups: int = 32):
        super().__init__()
        # in_channels + out_channels as we concatenate the output of the same resolution
        # of the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels, n_groups=n_groups)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)

        return x


class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, time_channels: int, n_groups: int = 32):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels, n_groups=n_groups)
        self.attn = SelfAttention(n_channels, n_heads=4)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels, n_groups=n_groups)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)

        return x


class UpSample(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)

        return x


class DownSample(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)

        return x


if __name__ == "__main__":
    bs = 16
    ipt = torch.randn((bs, 3, 32, 32))

    t_channels = 4
    time = torch.randn((bs, t_channels))

    # simulate U-Net
    b1 = DownBlock(3, 16, t_channels, n_groups=4)
    ipt = b1(ipt, time)
    print("Res Block:", ipt.shape)

    ds = DownSample(16)
    ipt = ds(ipt)
    print("Down sample:", ipt.shape)
