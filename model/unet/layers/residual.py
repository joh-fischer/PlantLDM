import torch
import torch.nn as nn
from model.unet.layers.swish import Swish


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_groups: int = 32):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.GroupNorm(n_groups, in_channels),
            Swish(beta=1.0),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(n_groups, out_channels),
            Swish(beta=1.0),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.time_emb = nn.Linear(time_channels, out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        identity = self.shortcut(x)
        x = self.block1(x)

        # condition with time
        t = self.time_emb(t)    # [bs, out_channels]
        x += t[:, :, None, None]

        x = self.block2(x)

        # shortcut connection
        x += identity

        return x


if __name__ == "__main__":
    bs = 8
    t_channels = 32

    ipt = torch.randn((bs, 64, 256, 256))
    time = torch.randn((bs, t_channels))

    res = ResidualBlock(64, 128, t_channels)
    out = res(ipt, time)

    print("Input:", ipt.shape)
    print("Time:", time.shape)
    print("Output:", out.shape)
