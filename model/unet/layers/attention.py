import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, d_k: int, n_heads: int = 2):
        """
        Applies self-attention like in "Attention Is All You Need" (https://arxiv.org/abs/1706.03762)
        to an image by reshaping it into a sequence. Only for small field sizes.

        Args:
            d_k (int): Dimension of queries, keys, and values
            n_heads: Number of heads for multi-head self attention
        """
        super().__init__()
        self.self_attention = nn.MultiheadAttention(d_k, n_heads)

    def forward(self, x: torch.Tensor):
        bs, n_channels, height, width = x.shape
        x = x.view(bs, n_channels, -1).permute(0, 2, 1)

        x, _ = self.self_attention(x, x, x)

        x = x.permute(0, 2, 1).view(bs, n_channels, height, width)

        return x


if __name__ == "__main__":
    ipt = torch.randn((4, 32, 16, 16))

    attention = SelfAttention(32, 2)

    out = attention(ipt)

    print("Input:", ipt.shape)
    print("Output:", out.shape)
