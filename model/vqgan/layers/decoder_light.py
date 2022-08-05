import torch
import torch.nn as nn

from model.layers import ResidualBlock
from model.layers import Attention


class DecoderLight(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int,
                 channels: list[int] = None,
                 dim_keys: int = 64, n_heads: int = 4):
        """
        Lightweight decoder for VQ-GAN with less parameters, converting
        a latent representation back to an image.

        Args:
            in_channels: Number of input channels of the image
            latent_dim: Number of channels for the latent space
            channels: List of channels for the number of down/up steps, eg [16, 32, 64]. Note
                that for the decoder the channels list will be reversed
            dim_keys: Dimension of keys, queries, values for attention layers
            n_heads: Number of heads for multi-head attention
        """
        super().__init__()
        self.channels = channels if channels is not None else [32, 64]
        self.n_blocks = len(self.channels)

        self.init_conv = nn.Sequential(
            nn.Conv2d(latent_dim, self.channels[-1], kernel_size=3, padding=1),
            nn.SiLU()
        )

        # bottleneck
        self.mid_block = ResidualBlock(self.channels[-1], self.channels[-1])
        self.mid_attn = Attention(self.channels[-1], dim_keys, n_heads)

        # decoder
        self.up_blocks = nn.ModuleList([])
        prev_channel = self.channels[-1]
        for c in reversed(self.channels):
            self.up_blocks.append(nn.Sequential(
                ResidualBlock(prev_channel, c),
                nn.ConvTranspose2d(c, c, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, True)
            ))
            prev_channel = c

        # output
        self.out = nn.Conv2d(self.channels[0], in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        # bottleneck
        x = self.init_conv(x)
        x = self.mid_block(x)
        x = self.mid_attn(x)

        for block in self.up_blocks:
            x = block(x)

        x = torch.tanh(self.out(x))

        return x


if __name__ == "__main__":
    ipt = torch.randn((8, 10, 32, 32))
    dec = DecoderLight(3, 10, [32, 64])
    out = dec(ipt)

    print("Input:", ipt.shape)      # [bs, 10, 32, 32]
    print("Output:", out.shape)     # [bs, 3, 128, 128]
