import torch
import torch.nn as nn

from model.losses.lpips import LPIPS


class LossVQGAN(nn.Module):
    def __init__(self,
                 perceptual_weight: int = 1,
                 codebook_weight: int = 1.,
                 commitment_weight: int = 0.25):
        """
        A class for computing and combining the different losses used in VQ-GAN.

        Args:
            perceptual_weight: Weight for the perceptual LPIPS loss.
            codebook_weight: Weight for the codebook loss of the vector quantizer.
            commitment_weight: Beta for the commitment loss of the vector quantizer.
        """
        super().__init__()

        self.perceptual_weight = perceptual_weight
        self.codebook_weight = codebook_weight
        self.commitment_weight = commitment_weight

        self.perceptual_loss = LPIPS().eval()

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor,
                z_e: torch.Tensor, z_q: torch.Tensor):
        """
        Computues the loss for VQ-GAN including the following sub-losses:
        - L1 reconstruction loss
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

        # reconstruction loss (take L1 as it is produces less blurry
        # results according to https://arxiv.org/abs/1611.07004)
        rec_loss = torch.mean(torch.abs(x_hat - x))

        # embedding loss (including stop-gradient according to
        # the paper https://arxiv.org/abs/1711.00937)
        embedding_loss = torch.mean((z_q.detach() - z_e) ** 2)
        commitment_loss = torch.mean((z_q - z_e.detach()) ** 2)
        codebook_loss = embedding_loss + self.commitment_weight * commitment_loss
        codebook_loss *= self.codebook_weight

        # perceptual loss (perceptual loss using LPIPS according
        # to the paper https://arxiv.org/abs/1801.03924)
        if self.perceptual_weight > 0:
            perceptual_loss = self.perceptual_loss(x_hat, x)
            perceptual_loss *= self.perceptual_weight
        else:
            perceptual_loss = torch.tensor([0.0])

        # TODO: disciminator loss

        # final loss and logs
        loss = rec_loss + codebook_loss + perceptual_loss
        log = {
            'rec_loss': rec_loss.item(),
            'codebook_loss': codebook_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
            'loss': loss.item()
        }

        return loss, log


if __name__ == "__main__":
    l_vqgan = LossVQGAN()

    ipt = torch.randn((8, 3, 128, 128))

    latent_e = torch.randn((8, 10, 32, 32))
    latent_q = torch.randn((8, 10, 32, 32))

    out, logs = l_vqgan(ipt, ipt, latent_e, latent_q)

    print("Output:", out)
    print("Logs:", logs)
