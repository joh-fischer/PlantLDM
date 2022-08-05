import torch
import torch.nn as nn


class CodebookLoss(nn.Module):
    def __init__(self, beta: float = 0.25):
        """
        Computes the codebook loss with stop-gradient operator, like
        specified in VQ-VAE paper (https://arxiv.org/abs/1711.00937)
        in equation (3).

        Args:
            beta: Scale factor for commitment loss (default and in
                paper: 0.25).
        """
        super().__init__()
        self.beta = beta

    def forward(self, z_e: torch.Tensor, z_q: torch.Tensor):
        """
        Computes the embedding loss, which optimizes the embeddings,
        and the commitment loss, which optimizes the encoder.

        Args:
            z_e: Encoded image.
            z_q: Quantized encoded image.

        Returns:
            codebook_loss: Sum of embedding and (scaled) commitment loss.
        """
        embedding_loss = torch.mean((z_q.detach() - z_e) ** 2)
        commitment_loss = torch.mean((z_q - z_e.detach()) ** 2)

        codebook_loss = embedding_loss + self.beta * commitment_loss

        return codebook_loss


if __name__ == "__main__":
    e = torch.randn((8, 10, 32, 32))
    z = torch.randn((8, 10, 32, 32))

    loss_fn = CodebookLoss()
    loss = loss_fn(e, z)
    print("Loss:", loss.item())
