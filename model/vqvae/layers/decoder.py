import torch
import torch.nn as nn
from model.vqvae.layers.residual import ResidualLayer


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int, n_res_layers: int = 2, res_hidden_dim: int = 256):
        """
        Decoder for VQ-VAE like specified in https://arxiv.org/abs/1711.00937. According to the paper
        'decoder has two residual 3 × 3 blocks, followed by two transposed convolutions with stride
        2 and window size 4 × 4' (paper p. 5).

        Args:
            latent_dim (int): Dimension of the latent space
            out_channels (int): Number of output channels
            n_res_layers (int): Number of residual layers (paper and default: 2)
            res_hidden_dim (int): Hidden dimension of residual blocks (paper and default: 256)
        """
        super(Decoder, self).__init__()

        kernel = 4
        stride = 2

        self.res_stack = nn.ModuleList([
            ResidualLayer(latent_dim, latent_dim, res_hidden_dim)
            for _ in range(n_res_layers)
        ])

        if latent_dim // 2 < 2:
            hidden_channels = 2
        else:
            hidden_channels = latent_dim // 2

        self.conv1 = nn.ConvTranspose2d(latent_dim, hidden_channels,
                                        kernel_size=kernel, stride=stride, padding=1)
        self.conv2 = nn.ConvTranspose2d(hidden_channels, out_channels,
                                        kernel_size=kernel, stride=stride, padding=1)

        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.res_stack:
            x = layer(x)
        x = self.activation(x)

        x = self.conv1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = torch.tanh(x)   # restrict to [-1, 1]

        return x


if __name__ == "__main__":
    latent = torch.randn((16, 10, 32, 32))
    dec = Decoder(10, 3)
    out = dec(latent)

    print("Input shape:", latent.shape)     # [bs, 10, 32, 32]
    print("Output shape:", out.shape)       # [bs, 3, 128, 128]
    print(f"\t {out.min()} < {torch.mean(out)} < {out.max()}")
