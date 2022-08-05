import torch
import torch.nn as nn

from model.layers import ResidualBlock
from model.layers import Attention


class EncoderLight(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int,
                 channels: list[int] = None,
                 dim_keys: int = 64, n_heads: int = 4):
        """
        Lightweight encoder for VQ-GAN with less parameters. The final
        latent resolution will be: img_size / 2^{len(channels)}.

        Args:
            in_channels: Number of input channels of the image
            latent_dim: Number of channels for the latent space
            channels: List of channels for the number of down/up steps, eg [16, 32, 64]
            dim_keys: Dimension of keys, queries, values for attention layers
            n_heads: Number of heads for multi-head attention
        """
        super().__init__()

        self.channels = channels if channels is not None else [32, 64]
        self.n_blocks = len(self.channels)

        # initial convolutional layer
        self.down_blocks = nn.ModuleList([])
        prev_channel = in_channels
        for c in self.channels:
            self.down_blocks.append(nn.Sequential(
                nn.Conv2d(prev_channel, c, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, True),
                ResidualBlock(c, c)
            ))
            prev_channel = c

        # bottleneck
        self.mid_attn = Attention(self.channels[-1], dim_keys, n_heads)
        self.mid_block = ResidualBlock(self.channels[-1], self.channels[-1])

        # output
        self.out = nn.Conv2d(self.channels[-1], latent_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        for block in self.down_blocks:
            x = block(x)

        # bottleneck
        x = self.mid_attn(x)
        x = self.mid_block(x)

        x = self.out(x)

        return x


if __name__ == "__main__":
    ipt = torch.randn((8, 3, 128, 128))
    enc = EncoderLight(3, 10, [32, 64])
    out = enc(ipt)

    print("Input:", ipt.shape)      # [bs, 3, 128, 128]
    print("Output:", out.shape)     # [bs, 10, 32, 32]
