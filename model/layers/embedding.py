import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, time_emb_dim: int, pos_emb_dim: int, max_len: int = 5000):
        """
        Time embedding for time step t. First, t is embedded using a fixed sinusoidal positional
        embedding, as described in "Attention Is All You Need" (https://arxiv.org/abs/1706.03762),
        followed by a two layer MLP.

        Args:
            time_emb_dim: Dimension of final time embedding
            pos_emb_dim: Embedding dimension for the fixed sinusoidal positional embedding
            max_len: Maximum number of time steps (default: 5000)
        """
        super().__init__()

        self.pos_emb_dim = pos_emb_dim
        self.time_emb_dim = time_emb_dim
        self.max_len = max_len

        # fixed sinusoidal positional embedding
        assert self.pos_emb_dim % 2 == 0, "Embedding dim must be a multiple of 2!"
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        _2i = torch.arange(0, self.pos_emb_dim, 2).float()
        pos_embedding = torch.zeros(self.max_len, self.pos_emb_dim)
        pos_embedding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / self.pos_emb_dim)))
        pos_embedding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / self.pos_emb_dim)))
        self.register_buffer('pos_embedding', pos_embedding, persistent=True)

        # MLP for time embedding
        self.mlp = nn.Sequential(
            nn.Linear(self.pos_emb_dim, self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )

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
    emb = TimeEmbedding(time_emb_dim=16, pos_emb_dim=4, max_len=10)
    time = torch.randint(0, 10, (4,))
    time_emb = emb.forward(time)

    print("Fixed positional embedding:\n", emb.pos_embedding)
    print("Time:", time)                        # eg [2, 5, 7, 9]
    print("Time embedding:", time_emb.shape)    # [4, 16]
