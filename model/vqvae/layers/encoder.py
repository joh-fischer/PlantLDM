import torch.nn as nn
from model.vqvae.layers.residual import ResidualLayer


class Encoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, n_res_layers: int = 2, res_hidden_dim: int = 256):
        """
        Encoder for VQ-VAE like specified in https://arxiv.org/abs/1711.00937. According to the paper
        'the encoder consists of 2 strided convolutional layers with stride 2 and window size 4 × 4,
        followed by two residual 3 × 3 blocks (implemented as ReLU, 3x3 conv, ReLU, 1x1 conv), all
        having 256 hidden units' (paper p. 5).

        Args:
            in_channels (int): Number of input channels
            latent_dim (int): Dimension of the latent space
            n_res_layers (int): Number of residual layers (paper and default: 2)
            res_hidden_dim (int): Hidden dimension of residual blocks (paper and default: 256)
        """
        super(Encoder, self).__init__()

        kernel = 4
        stride = 2

        if latent_dim // 2 < 2:
            hidden_channels = 2
        else:
            hidden_channels = latent_dim // 2

        self.conv1 = nn.Conv2d(in_channels, hidden_channels,
                               kernel_size=kernel, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, latent_dim,
                               kernel_size=kernel, stride=stride, padding=1)

        self.res_stack = nn.ModuleList([
            ResidualLayer(latent_dim, latent_dim, res_hidden_dim)
            for _ in range(n_res_layers)
        ])

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)

        x = self.conv2(x)

        for layer in self.res_stack:
            x = layer(x)
        x = self.activation(x)

        return x


if __name__ == "__main__":
    import torch

    ipt = torch.randn((16, 3, 128, 128))
    enc = Encoder(3, 10)
    out = enc(ipt)

    print("Input shape:", ipt.shape)    # [bs, 3, 128, 128]
    print("Output shape:", out.shape)   # [bs, 10, 32, 32]
