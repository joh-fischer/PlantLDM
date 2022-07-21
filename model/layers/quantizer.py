import torch
import torch.nn as nn


class VectorQuantizer(nn.Module):
    def __init__(self, n_embeddings: int, embedding_dim: int):
        """
        Vector quantizer that discretizes the continuous latent z.

        Args:
            n_embeddings (int): Codebook size
            embedding_dim (int): Dimension of the latent z (channels)
        """
        super(VectorQuantizer, self).__init__()

        self.n_emb = n_embeddings
        self.e_dim = embedding_dim

        self.embedding = nn.Embedding(self.n_emb, self.e_dim)
        self.embedding.weight.data.uniform_(-1. / self.e_dim, 1. / self.e_dim)

    def forward(self, z):
        """
        Maps the output of the encoder network z (continuous) to a discrete one-hot
        vector z_q, where the index indicates the closest embedding vector e_j. The
        latent z is detached as first step to allow straight through backprop.

        Args:
            z: Output of the encoder network, shape [bs, latent_dim, h, w]

        Returns:

        """
        z = z.detach()
        # flatten input from [bs, c, h, w] to [bs*h*w, c]
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.e_dim)

        # calculate distances between each z [bs*h*w, c]
        # and e_j [n_emb, c]: (z - e_j)² = z² + e² - e*z*2
        # TODO: maybe make more intuitive with (z - e_j)²
        z_sq = torch.sum(z_flat**2, dim=1, keepdim=True)
        e_sq = torch.sum(self.embedding.weight**2, dim=1)
        e_z = torch.matmul(z_flat, self.embedding.weight.t())
        dists = z_sq + e_sq - 2 * e_z    # [bs*h*w, n_emb]

        # get index of the closest embedding e_j for each vector z
        argmin_inds = torch.argmin(dists, dim=1)

        # one-hot encode
        argmin_one_hot = nn.functional.one_hot(argmin_inds, num_classes=self.n_emb)
        # ensure correct device and float dtype for matmul
        argmin_one_hot = argmin_one_hot.float().to(z.device)

        # multiply one-hot w. embedding weights to get quantized z
        z_q = torch.matmul(argmin_one_hot, self.embedding.weight)
        # reshape to [bs, c, h, w]
        z_q = z_q.view(z.shape).permute(0, 3, 1, 2)

        return z_q


if __name__ == "__main__":
    # encoder output and decoder input: [bs, 10, 32, 32]
    latent = torch.randn((8, 10, 32, 32))

    vq = VectorQuantizer(4, 10)
    out = vq(latent)

    print("Input shape:", latent.shape)
    print("Output shape:", out.shape)
