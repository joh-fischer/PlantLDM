import torch
import torch.nn as nn


class ReconstructionLoss(nn.Module):
    def __init__(self, loss_type: str):
        """
        Computes the reconstruction loss.

        Args:
            loss_type: Either L1 or L2.
        """
        super().__init__()

        if loss_type.lower() == 'l1':
            self.rec_loss_fn = torch.nn.functional.l1_loss
        elif loss_type.lower() == 'l2':
            self.rec_loss_fn = torch.nn.functional.mse_loss
        else:
            raise ValueError(f"Unknown reconstruction loss type '{loss_type}'!")

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor):
        rec_loss = self.rec_loss_fn(x_hat, x)

        return rec_loss


if __name__ == "__main__":
    ipt = torch.randn((8, 3, 32, 32))
    tgt = torch.randn((8, 3, 32, 32))

    loss_fn = ReconstructionLoss('L1')
    print("Diff:", loss_fn(ipt, tgt).item())
    print("Same:", loss_fn(ipt, ipt).item())
