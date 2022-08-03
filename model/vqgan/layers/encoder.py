import torch
import torch.nn as nn

from model.layers.residual import ResidualBlock
from model.layers.scaling import DownSample
from model.layers.attention import LinearAttention, Attention


class Encoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int,
                 channels=None,
                 dim_keys: int = 64, n_heads: int = 4):
        """
        Encoder for VQ-GAN like specified in https://arxiv.org/abs/2012.09841, table 7. The
        final latent resolution will be: img_size / 2^{len(channels)}.

        Args:
            in_channels: Number of input channels of the image
            latent_dim: Number of channels for the latent space
            channels: List of channels for the number of down/up steps, eg [16, 32, 64]
            dim_keys: Dimension of keys, queries, values for attention layers
            n_heads: Number of heads for multi-head attention
        """
        super().__init__()

        self.channels = channels if channels is not None else [16, 32, 64]
        self.n_blocks = len(self.channels)

        # initial convolutional layer
        self.init_conv = nn.Conv2d(in_channels, self.channels[0], kernel_size=7, padding=3)

        # encoder
        self.down_blocks = nn.ModuleList([])
        prev_channel = self.channels[0]
        for c in self.channels:
            self.down_blocks.append(
                nn.ModuleList([
                    ResidualBlock(prev_channel, c),
                    ResidualBlock(c, c),
                    nn.GroupNorm(1, c),
                    LinearAttention(c, dim_keys, n_heads),
                    DownSample(c)
                ])
            )
            prev_channel = c

        # bottleneck
        self.mid_block1 = ResidualBlock(self.channels[-1], self.channels[-1])
        self.mid_attn = Attention(self.channels[-1], dim_keys, n_heads)
        self.mid_block2 = ResidualBlock(self.channels[-1], self.channels[-1])

        # output
        self.out = nn.Conv2d(self.channels[-1], latent_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        x = self.init_conv(x)

        for block1, block2, norm, attn, downsample in self.down_blocks:
            x = block1(x)
            x = block2(x)
            x = norm(x)
            x = attn(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        x = self.out(x)

        return x


if __name__ == "__main__":
    ipt = torch.randn((8, 3, 128, 128))
    enc = Encoder(3, 10, [32, 64, 128])
    out = enc(ipt)

    print("Input:", ipt.shape)      # [bs, 3, 128, 128]
    print("Output:", out.shape)     # [bs, 10, 16, 16]
