import torch.nn as nn

from model.vqvae.layers import Encoder
from model.vqvae.layers.decoder import Decoder
from model.vqvae.layers.quantizer import VectorQuantizer


class VQVAE(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int,
                 n_res_layers: int = 2, res_hidden_dim: int = 256,
                 n_embeddings: int = 512):
        super(VQVAE, self).__init__()

        self.encoder = Encoder(in_channels, latent_dim, n_res_layers, res_hidden_dim)

        self.vq = VectorQuantizer(n_embeddings, latent_dim)

        self.decoder = Decoder(latent_dim, in_channels, n_res_layers, res_hidden_dim)

    def forward(self, x):
        z_e = self.encoder(x)

        z_q = self.vq(z_e)

        x_hat = self.decoder(z_q)

        return x_hat, z_e, z_q

    def encode(self, x):
        z_e = self.encoder(x)
        z_q = self.vq(z_e)

        return z_q

    def decode(self, z_e):
        z_q = self.vq(z_e)
        x_hat = self.decoder(z_q)

        return x_hat


if __name__ == "__main__":
    import torch

    ipt = torch.randn((16, 3, 128, 128))

    vqvae = VQVAE(3, 10)
    rec, e, q = vqvae(ipt)

    print("Input shape:", ipt.shape)    # [bs, 3, 128, 128]
    print("rec shape:", rec.shape)      # [bs, 3, 128, 128]
    print("encoded:", e.shape)
    print("quantized:", q.shape)
