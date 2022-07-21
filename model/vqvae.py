import torch.nn as nn

from model.layers.encoder import Encoder
from model.layers.decoder import Decoder
from model.layers.quantizer import VectorQuantizer


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

        z_q, z_q_sg = self.vq(z_e)

        x_hat = self.decoder(z_q_sg)

        return z_e, z_q, z_q_sg, x_hat


if __name__ == "__main__":
    import torch

    ipt = torch.randn((16, 3, 128, 128))

    vqvae = VQVAE(3, 10)
    z, q, q_sg, rec = vqvae(ipt)

    print("Input shape:", ipt.shape)    # [bs, 3, 128, 128]
    print("z_e shape:", z.shape)        # [bs, 10, 32, 32]
    print("z_q shape:", q.shape)        # [bs, 10, 32, 32]
    print("z_q sg shape:", q_sg.shape)  # [bs, 10, 32, 32]
    print("rec shape:", rec.shape)      # [bs, 3, 128, 128]
