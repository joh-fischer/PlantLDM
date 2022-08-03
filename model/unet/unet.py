import torch
import torch.nn as nn

from model.layers import UpSample, DownSample
from model.layers import LinearAttention, Attention
from model.layers import TimeEmbedding
from model.layers import ResidualBlock


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int, time_emb_dim: int, pos_emb_dim: int,
                 channels: list[int] = None,
                 n_groups: int = 8, dim_keys: int = 64, n_heads: int = 4):
        """
        U-Net model, first proposed in (https://arxiv.org/abs/1505.04597) and equipped for
        our DDPM with (linear) attention and time conditioning.

        Args:
            in_channels: Channels of the input image
            time_emb_dim: Dimension of time embedding
            pos_emb_dim: Dimension of fixed sinusoidal positional embedding
            channels: List of channels for the number of down/up steps
            n_groups: Number of groups for group normalization
            dim_keys: Dimension of keys, queries, values for attention layers
            n_heads: Number of heads for multi-head attention
        """
        super().__init__()

        self.channels = channels if channels is not None else [16, 32, 64]
        self.n_blocks = len(self.channels)

        # time embedding
        self.time_embedding = TimeEmbedding(time_emb_dim, pos_emb_dim)

        # initial convolutional layer
        self.init_conv = nn.Conv2d(in_channels, self.channels[0], kernel_size=7, padding=3)

        # contracting path
        self.down_blocks = nn.ModuleList([])
        prev_channel = self.channels[0]
        for c in self.channels:
            self.down_blocks.append(
                nn.ModuleList([
                    ResidualBlock(prev_channel, c, time_emb_dim, n_groups),
                    ResidualBlock(c, c, time_emb_dim, n_groups),
                    nn.GroupNorm(1, c),
                    LinearAttention(c, dim_keys, n_heads),
                    DownSample(c)
                ])
            )
            prev_channel = c

        # bottleneck
        self.mid_block1 = ResidualBlock(self.channels[-1], self.channels[-1], time_emb_dim, n_groups)
        self.mid_attn = Attention(self.channels[-1], dim_keys, n_heads)
        self.mid_block2 = ResidualBlock(self.channels[-1], self.channels[-1], time_emb_dim, n_groups)

        # expanding path
        self.up_blocks = nn.ModuleList([])
        prev_channel = self.channels[-1]
        for c in reversed(self.channels):
            self.up_blocks.append(
                nn.ModuleList([
                    UpSample(prev_channel),
                    ResidualBlock(prev_channel + c, c, time_emb_dim, n_groups),
                    ResidualBlock(c, c, time_emb_dim, n_groups),
                    nn.GroupNorm(1, c),
                    LinearAttention(c, dim_keys, n_heads)
                ])
            )
            prev_channel = c

        # final output 1x1 convolution
        self.final_conv = nn.Conv2d(self.channels[0], in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t = self.time_embedding(t)
        x = self.init_conv(x)

        skips = []

        # down sample
        for block1, block2, norm, attn, downsample in self.down_blocks:
            x = block1(x, t)
            x = block2(x, t)
            x = norm(x)
            x = attn(x)
            skips.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # up sample
        for upsample, block1, block2, norm, attn in self.up_blocks:
            x = upsample(x)
            x = torch.cat((x, skips.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = norm(x)
            x = attn(x)

        # output convolution
        x = self.final_conv(x)

        return x


if __name__ == "__main__":
    t_emb_dim = 16
    p_emb_dim = 8

    u = UNet(3, t_emb_dim, p_emb_dim, channels=[16, 32, 64, 128])

    ipt = torch.randn((8, 3, 128, 128))
    time = torch.randint(0, 10, (8,))

    out = u(ipt, time)
    print("Input:", ipt.shape)
    print("Output:", out.shape)
