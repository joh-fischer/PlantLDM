import torch
import torch.nn as nn

from model.layers import VectorQuantizer
from model.vqgan.layers import EncoderLight
from model.vqgan.layers import DecoderLight


class VQGANLight(nn.Module):
    def __init__(self, latent_dim: int, autoencoder_cfg: dict, n_embeddings: int = 512):
        """
        Lightweight Vector-quantized GAN (paper: https://arxiv.org/abs/2012.09841)
        with fewer parameters in the encoder/decoder.

        Args:
            latent_dim: Latent dimension of the embedding/codebook
            autoencoder_cfg: Dictionary containing the information for the encoder and decoder. For
                example {'in_channels': 3, 'channels': [16, 32, 64], 'dim_keys': 64, 'n_heads': 4}.
            n_embeddings: Number of embeddings for the codebook
        """
        super().__init__()
        self.encoder = EncoderLight(latent_dim=latent_dim, **autoencoder_cfg)
        self.vq = VectorQuantizer(n_embeddings, latent_dim)
        self.decoder = DecoderLight(latent_dim=latent_dim, **autoencoder_cfg)

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

    autoencoder_cfg = {
        'in_channels': 3,
        'channels': [32, 64],
        'dim_keys': 64,
        'n_heads': 4
    }
    vqgan = VQGANLight(10, autoencoder_cfg)
    rec, e, q = vqgan(ipt)

    print("Input shape:", ipt.shape)    # [bs, 3, 128, 128]
    print("rec shape:", rec.shape)      # [bs, 3, 128, 128]
    print("embedding:", e.shape)        # [bs, 10, 32, 32]
    print("quantized:", q.shape)        # [bs, 10, 32, 32]
