import torch
import torch.nn as nn

from model.vqvae.layers import Encoder
from model.vqvae.layers import Decoder
from model.layers import VectorQuantizer


class VQVAE(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int,
                 n_res_layers: int = 2, res_hidden_dim: int = 256,
                 n_embeddings: int = 512):
        """
        Vector-Quantized Variational Autoencoder (paper: https://arxiv.org/abs/1711.00937).

        Args:
            in_channels: Image input channels
            latent_dim: Latent dimension of the embedding/codebook
            n_res_layers: Number of residual blocks
            res_hidden_dim: Hidden dimension of the residual blocks
            n_embeddings: Number of embeddings for the codebook
        """
        super(VQVAE, self).__init__()

        self.encoder = Encoder(in_channels, latent_dim, n_res_layers, res_hidden_dim)

        self.vq = VectorQuantizer(n_embeddings, latent_dim)

        self.decoder = Decoder(latent_dim, in_channels, n_res_layers, res_hidden_dim)

    def forward(self, x: torch.Tensor):
        """ Forward pass through vector-quantized variational autoencoder.

        Args:
            x: Input image tensor.
        Returns:
            x_hat: Reconstructed image x
            z_e: Latent (un-quantized) representation of image x
            z_q: Quantized latent representation of image x
        """
        z_e = self.encoder(x)

        z_q = self.vq(z_e)

        # preserve gradients
        z_q_ = z_e + (z_q - z_e).detach()

        x_hat = self.decoder(z_q_)

        return x_hat, z_e, z_q

    def encode(self, x: torch.Tensor):
        """ Encode input image.

        Args:
            x: Input image
        Returns:
            z_e: Encoded input image (un-quantized).
        """
        z_e = self.encoder(x)

        return z_e

    def quantize(self, z_e: torch.Tensor):
        """ Quantize latent representation.

        Args:
            z_e: Un-quantized latent representation (encoded image).
        Returns:
            z_q: Quantized embedding.
        """
        z_q = self.vq(z_e)

        return z_q

    def decode(self, z_e: torch.Tensor):
        """ Decode latent representation to input image.

        Args:
            z_e: Un-quantized latent representation.
        Returns:
            x_hat: Reconstructed input image.
        """
        z_q = self.vq(z_e)
        x_hat = self.decoder(z_q)

        return x_hat


if __name__ == "__main__":
    ipt = torch.randn((16, 3, 128, 128))

    vqvae = VQVAE(3, 10)
    rec, e, q = vqvae(ipt)

    print("Input shape:", ipt.shape)    # [bs, 3, 128, 128]
    print("rec shape:", rec.shape)      # [bs, 3, 128, 128]
    print("embedding:", e.shape)        # [bs, 10, 32, 32]
    print("quantized:", q.shape)        # [bs, 10, 32, 32]
