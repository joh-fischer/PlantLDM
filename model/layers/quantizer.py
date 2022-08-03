import torch
import torch.nn as nn
import einops


class VectorQuantizer(nn.Module):
    def __init__(self, n_embeddings: int, latent_dim: int):
        """
        Vector quantizer that discretizes the continuous latent z. Adapted from
        https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py.
        Args:
            n_embeddings (int): Codebook size
            latent_dim (int): Dimension of the latent z (channels)
        """
        super(VectorQuantizer, self).__init__()

        self.n_emb = n_embeddings
        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(self.n_emb, self.latent_dim)
        self.embedding.weight.data.uniform_(-1. / self.latent_dim, 1. / self.latent_dim)

    def forward(self, z: torch.Tensor):
        """
        Maps the output of the encoder network z (continuous) to a discrete one-hot
        vector z_q, where the index indicates the closest embedding vector e_j. The
        latent z is detached as first step to allow straight through backprop.
        Args:
            z: Output of the encoder network, shape [bs, latent_dim, h, w]
        Returns:
            z_q: Quantized z
        """
        bs, c, h, w = z.shape

        # flatten input from [bs, c, h, w] to [bs*h*w, c]
        z_flat = einops.rearrange(z, 'b c h w -> (b h w) c')

        # calculate distances between each z [bs*h*w, c]
        # and e_j [n_emb, c]: (z - e_j)² = z² + e² - e*z*2
        z_sq = torch.sum(z_flat**2, dim=1, keepdim=True)
        e_sq = torch.sum(self.embedding.weight**2, dim=1)
        e_z = torch.matmul(z_flat, self.embedding.weight.t())
        distances = z_sq + e_sq - 2 * e_z    # [bs*h*w, n_emb]

        # get index of the closest embedding e_j for each vector z
        argmin_inds = torch.argmin(distances, dim=1)

        # one-hot encode
        argmin_one_hot = nn.functional.one_hot(argmin_inds, num_classes=self.n_emb).float().to(z.device)

        # multiply one-hot w. embedding weights to get quantized z
        z_q = torch.matmul(argmin_one_hot, self.embedding.weight)

        # reshape back to [bs, c, h, w]
        z_q = einops.rearrange(z_q, '(b h w) c -> b c h w', b=bs, h=h, w=w)

        return z_q


if __name__ == "__main__":
    latent = torch.randn((8, 10, 32, 32))

    vq = VectorQuantizer(4, 10)
    q = vq(latent)

    print("Input shape:", latent.shape)
    print("z_q shape:", q.shape)
