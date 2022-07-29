import torch
import torch.nn as nn


class VectorQuantizer(nn.Module):
    def __init__(self, n_embeddings: int, latent_dim: int, beta: float = 0.25):
        """
        Vector quantizer that discretizes the continuous latent z. Adapted from
        https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py.

        Args:
            n_embeddings (int): Codebook size
            latent_dim (int): Dimension of the latent z (channels)
            beta (float): Factor for commitment loss
        """
        super(VectorQuantizer, self).__init__()

        self.n_emb = n_embeddings
        self.latent_dim = latent_dim
        self.beta = beta

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
            loss: Sum of embedding loss and scaled commitment loss
        """
        # flatten input from [bs, c, h, w] to [bs*h*w, c]
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.latent_dim)

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
        z_q = torch.matmul(argmin_one_hot, self.embedding.weight).view(z.shape)

        # compute loss (embedding & commitment)
        embedding_loss = torch.mean((z_q.detach() - z)**2)
        commitment_loss = self.beta * torch.mean((z_q - z.detach())**2)
        loss = embedding_loss + commitment_loss

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to [bs, c, h, w]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss


if __name__ == "__main__":
    latent = torch.randn((8, 10, 32, 32))

    vq = VectorQuantizer(4, 10)
    q, out_loss = vq(latent)

    print("Input shape:", latent.shape)
    print("z_q shape:", q.shape)
    print("loss:", out_loss.item())
