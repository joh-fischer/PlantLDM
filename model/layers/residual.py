import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int = None, n_groups: int = 8):
        """
        Residual block with time conditioning.

        Args:
            in_channels: Input channels to residual block
            out_channels: Output channels of residual block
            time_emb_dim: Dimension of time embedding
            n_groups: Number of groups for group normalization
        """
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(n_groups, out_channels),
            nn.SiLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(n_groups, out_channels),
            nn.SiLU()
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.time_emb = nn.Linear(time_emb_dim, out_channels) if time_emb_dim is not None else None

    def forward(self, x: torch.Tensor, t: torch.Tensor = None):
        identity = self.shortcut(x)

        x = self.block1(x)

        # condition with time if required
        if self.time_emb is not None:
            t = self.time_emb(t)        # [bs, out_channels]
            x += t[:, :, None, None]

        x = self.block2(x)

        return x + identity


if __name__ == "__main__":
    bs = 8
    t_channels = 16

    ipt = torch.randn((bs, 64, 32, 32))
    time = torch.randn((bs, t_channels))

    res = ResidualBlock(64, 128, t_channels)
    out = res(ipt, time)

    print("Input:", ipt.shape)
    print("Time:", time.shape)
    print("Output:", out.shape)
