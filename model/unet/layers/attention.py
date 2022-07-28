import torch
import torch.nn as nn
from einops import rearrange


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


class LinearAttention(nn.Module):
    def __init__(self, n_channels, n_heads: int = 4, d_k: int = 32):
        """
        Efficient Attention (https://arxiv.org/abs/1812.01243), which instead of
        computing V (Q K.T) like in dot-product attention, computes Q (K.T V).
        This results in less complexity, O(d_k * d_v) instead of O(n²).

        Args:
            n_channels (int): Number of channels of the input feature maps
            n_heads (int): Number of heads for attention
            d_k (int): Dimension of queries, keys, and values
        """
        super().__init__()
        self.scale = d_k ** -0.5
        self.n_heads = n_heads
        hidden_dim = d_k * n_heads
        self.to_qkv = nn.Conv2d(n_channels, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, n_channels, 1),
                                    nn.GroupNorm(1, n_channels))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.n_heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        res = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        res = rearrange(res, "b h c (x y) -> b (h c) x y", h=self.n_heads, x=h, y=w)

        return self.to_out(res)


if __name__ == "__main__":
    ipt = torch.randn((4, 32, 16, 16))

    self_attn = SelfAttention(d_k=32, n_heads=2)
    out = self_attn(ipt)
    print("Self Attention")
    print("\tInput:", ipt.shape)
    print("\tOutput:", out.shape)

    lin_attn = LinearAttention(32, n_heads=2, d_k=32)
    out = lin_attn(ipt)
    print("Linear Attention")
    print("\tInput:", ipt.shape)
    print("\tOutput:", out.shape)
