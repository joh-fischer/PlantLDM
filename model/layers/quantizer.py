import torch
import torch.nn as nn


class VectorQuantizer(nn.Module):
    def __init__(self, n_embeddings: int, embedding_dim: int):
        super(VectorQuantizer, self).__init__()

        self.n_emb = n_embeddings
        self.e_dim = embedding_dim

        self.embedding = nn.Embedding(self.n_emb, self.e_dim)
        self.embedding.weight.data.uniform_(-1. / self.e_dim, 1. / self.e_dim)

    def forward(self, z):
        """
        Maps the output of the encoder network z (continuous) to a discrete one-hot
        vector z_q, where the index indicates the closest embedding vector e_j.

        Args:
            z: Output of the encoder network, shape [bs, latent_dim, h, w]

        Returns:

        """
        # flatten input from [bs, c, h, w] to [bs*h*w, c]
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.e_dim)

        # calculate distances between each z [bs*h*w, c]
        # and e_j [n_emb, c]: (z - e_j)² = z² + e² - e*z*2
        # TODO: maybe make more intuitive
        z_sq = torch.sum(z_flat**2, dim=1, keepdim=True)
        e_sq = torch.sum(self.embedding.weight**2, dim=1)
        e_z = torch.matmul(z_flat, self.embedding.weight.t())

        dist = z_sq + e_sq - 2 * e_z    # [bs*h*w, n_emb]

        # get index of the closest embedding e_j for each vector z
        argmin_j = torch.argmin(dist, dim=1)

        # one-hot encode
        

