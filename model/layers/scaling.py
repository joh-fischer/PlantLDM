import torch
import torch.nn as nn


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
    bs = 16

    ipt = torch.randn((16, 32, 128, 128))
    print("Input:", ipt.shape)

    ds = DownSample(32)
    out = ds(ipt)
    print("Downsampled:", out.shape)

    us = UpSample(32)
    out = us(out)
    print("Upsampled:", out.shape)
