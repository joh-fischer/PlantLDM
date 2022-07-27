import torch
import torch.nn as nn
from model.unet.layers import Swish


class TimeEmbedding(nn.Module):
    def __init__(self, n_channels: int, embedding_dim: int, max_len: int = 5000):
        """
        Time embedding for time step t. First, t is embedded using a fixed sinusoidal positional
        embedding, as described in "Attention Is All You Need" (https://arxiv.org/abs/1706.03762),
        followed by a two layer MLP.

        Args:
            n_channels: Dimension of final time embedding
            embedding_dim: Embedding dimension for the fixed sinusoidal positional embedding
            max_len: Maximum number of time steps (default: 5000)
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_channels = n_channels
        self.max_len = max_len

        # fixed sinusoidal positional embedding
        assert self.embedding_dim % 2 == 0, "Embedding dim must be a multiple of 2!"
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        _2i = torch.arange(0, self.embedding_dim, 2).float()
        pos_embedding = torch.zeros(self.max_len, self.embedding_dim)
        pos_embedding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / self.embedding_dim)))
        pos_embedding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / self.embedding_dim)))
        self.register_buffer('pos_embedding', pos_embedding, persistent=True)

        # MLP for time embedding
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, self.n_channels),
            Swish(beta=1.0),
            nn.Linear(self.n_channels, self.n_channels)
        )

        print(self.pos_embedding.device)

    def forward(self, t: torch.Tensor):
        """
        Embeds a time step t with a fixed sinusoidal positional encoding, and then
        transforms it with an MLP.

        Args:
            t (torch.Tensor): Batch of time steps with shape [bs,]

        Returns:
            t_emb (torch.Tensor): Time embeddings for the respective time indices
        """
        t_pos_emb = torch.index_select(self.pos_embedding, 0, t)
        t_emb = self.mlp(t_pos_emb)

        return t_emb


if __name__ == "__main__":
    emb = TimeEmbedding(16, embedding_dim=4, max_len=10)
    print("Fixed positional embedding:\n", emb.pos_embedding)

    time = torch.randint(0, 10, (4,))
    print("Time:", time)       # eg [2, 5, 7, 9]

    time_emb = emb.forward(time)
    print("Time embedding:", time_emb.shape)    # [4, 16]
