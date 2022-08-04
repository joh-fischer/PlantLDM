import torch
import torch.nn as nn

from model.losses.lpips import LPIPS
from model.losses.reconstruction import ReconstructionLoss
from model.losses.codebook import CodebookLoss


class LossFn(nn.Module):
    def __init__(self,
                 rec_loss_type: str = 'L1',
                 perceptual_weight: int = 1,
                 codebook_weight: int = 1.,
                 commitment_weight: int = 0.25):
        """
        A class for computing and combining the different losses used in VQ-VAE
        and VQ-GAN.

        Args:
            rec_loss_type: Loss-type for reconstruction loss, either L1 or L2.
            perceptual_weight: Weight for the perceptual LPIPS loss.
            codebook_weight: Weight for the codebook loss of the vector quantizer.
            commitment_weight: Beta for the commitment loss of the vector quantizer.
        """
        super().__init__()
        # reconstruction loss (take L1 as it is produces less blurry
        # results according to https://arxiv.org/abs/1611.07004)
        self.rec_loss_fn = ReconstructionLoss(rec_loss_type)

        # embedding loss (including stop-gradient according to
        # the paper https://arxiv.org/abs/1711.00937)
        self.codebook_weight = codebook_weight
        self.codebook_loss_fn = CodebookLoss(commitment_weight)

        # perceptual loss (perceptual loss using LPIPS according
        # to the paper https://arxiv.org/abs/1801.03924)
        self.perceptual_weight = perceptual_weight
        self.perceptual_loss_fn = LPIPS().eval() if perceptual_weight > 0 else None

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor,
                z_e: torch.Tensor, z_q: torch.Tensor):
        """
        Computes the final loss including the following sub-losses:
        - reconstruction loss
        - codebook loss for vector quantizer
        - perceptual loss using LPIPS

        Args:
            x_hat: Reconstructed image.
            x: Original image.
            z_e: Encoded image.
            z_q: Quantized encoded image.

        Returns:
            loss: The combined loss.
            log: A dictionary containing all sub-losses and the total loss.
        """
        log = {}
        loss = torch.tensor([0.0]).to(x.device)

        rec_loss = self.rec_loss_fn(x_hat, x)
        loss += rec_loss
        log['rec_loss'] = rec_loss.item()

        codebook_loss = self.codebook_weight * self.codebook_loss_fn(z_e, z_q)
        loss += codebook_loss
        log['codebook_loss'] = codebook_loss.item()

        if self.perceptual_loss_fn is not None:
            perceptual_loss = self.perceptual_loss_fn(x_hat, x)
            perceptual_loss *= self.perceptual_weight
            loss += perceptual_loss
            log['perceptual_loss'] = perceptual_loss.item()

        log['loss'] = loss.item()

        return loss, log


if __name__ == "__main__":
    l_vqgan = LossFn(perceptual_weight=1)

    ipt = torch.randn((8, 3, 128, 128))

    latent_e = torch.randn((8, 10, 32, 32))
    latent_q = torch.randn((8, 10, 32, 32))

    out, logs = l_vqgan(ipt, ipt, latent_e, latent_q)

    print("Output:", out)
    print("Logs:", logs)
