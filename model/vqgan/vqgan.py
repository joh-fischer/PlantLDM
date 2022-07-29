import torch
import torch.nn as nn

from model.layers.quantizer import VectorQuantizer
from model.vqgan.layers.encoder import Encoder
from model.vqgan.layers.decoder import Decoder


class VQGAN(nn.Module):
    def __init__(self, latent_dim: int,
                 ae_cfg: dict,
                 n_embeddings: int = 512):
        super().__init__()

        self.encoder = Encoder(latent_dim=latent_dim, **ae_cfg)

        self.vq = VectorQuantizer(n_embeddings, latent_dim)

        self.decoder = Decoder(latent_dim=latent_dim, **ae_cfg)

    def forward(self, x: torch.Tensor):
        z_e = self.encoder(x)

        z_q, loss = self.vq(z_e)

        x_hat = self.decoder(z_q)

        return x_hat, loss

    def encode(self, x: torch.Tensor):
        z_e = self.encoder(x)
        z_q, _ = self.vq(z_e)

        return z_q

    def decode(self, z_e: torch.Tensor):
        z_q, _ = self.vq(z_e)
        x_hat = self.decoder(z_q)

        return x_hat


if __name__ == "__main__":
    ipt = torch.randn((16, 3, 128, 128))

    autoencoder_cfg = {
        'in_channels': 3,
        'channels': [16, 32, 64],
        'dim_keys': 64,
        'n_heads': 4
    }

    vqgan = VQGAN(10, autoencoder_cfg)
    rec, out_loss = vqgan(ipt)

    print("Input shape:", ipt.shape)    # [bs, 3, 128, 128]
    print("rec shape:", rec.shape)      # [bs, 3, 128, 128]
    print("Loss:", out_loss)
