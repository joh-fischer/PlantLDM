import torch
import torch.nn as nn

from model.layers.residual import ResidualBlock
from model.layers.scaling import UpSample
from model.layers.attention import LinearAttention, Attention


class Decoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int,
                 channels: list[int] = None,
                 dim_keys: int = 64, n_heads: int = 4):
        """
        Decoder for VQ-GAN like specified in https://arxiv.org/abs/2012.09841, table 7, converting
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

        self.channels = channels if channels is not None else [16, 32, 64]
        self.n_blocks = len(self.channels)

        self.init_conv = nn.Conv2d(latent_dim, self.channels[-1], kernel_size=3, padding=1)

        # bottleneck
        self.mid_block1 = ResidualBlock(self.channels[-1], self.channels[-1])
        self.mid_attn = Attention(self.channels[-1], dim_keys, n_heads)
        self.mid_block2 = ResidualBlock(self.channels[-1], self.channels[-1])

        # decoder
        self.up_blocks = nn.ModuleList([])
        prev_channel = self.channels[-1]
        for c in reversed(self.channels):
            self.up_blocks.append(
                nn.ModuleList([
                    UpSample(prev_channel),
                    ResidualBlock(prev_channel, c),
                    ResidualBlock(c, c),
                    nn.GroupNorm(1, c),
                    LinearAttention(c, dim_keys, n_heads)
                ])
            )
            prev_channel = c

        # output
        self.out = nn.Conv2d(self.channels[0], in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        # bottleneck
        x = self.init_conv(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        for upsample, block1, block2, norm, attn in self.up_blocks:
            x = upsample(x)
            x = block1(x)
            x = block2(x)
            x = norm(x)
            x = attn(x)

        x = torch.tanh(self.out(x))

        return x


if __name__ == "__main__":
    ipt = torch.randn((8, 10, 16, 16))
    dec = Decoder(3, 10, [32, 64, 128])
    out = dec(ipt)

    print("Input:", ipt.shape)      # [bs, 10, 16, 16]
    print("Output:", out.shape)     # [bs, 3, 128, 128]
