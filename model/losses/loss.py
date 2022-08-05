import torch
import torch.nn as nn

from model.losses.lpips import LPIPS
from model.losses.reconstruction import ReconstructionLoss
from model.losses.codebook import CodebookLoss
from model.losses.discriminator import Discriminator


class LossFn(nn.Module):
    def __init__(self,
                 rec_loss_type: str = 'L1',
                 perceptual_weight: float = 1,
                 codebook_weight: float = 1.,
                 commitment_weight: float = 0.25,
                 disc_weight: float = 1.,
                 disc_in_channels: int = 3,
                 disc_n_layers: int = 2,
                 disc_res_blocks: bool = False):
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

        # discriminator loss to avoid blurry images, as they
        # are classified as fake as long as they are blurry
        # (according to https://arxiv.org/abs/1611.07004)
        self.disc_weight = disc_weight
        self.discriminator = Discriminator(disc_in_channels, n_layers=disc_n_layers,
                                           residual_blocks=disc_res_blocks) if disc_weight > 0 else None

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor,
                z_e: torch.Tensor, z_q: torch.Tensor,
                train_autoencoder: bool = True):
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
            train_autoencoder: If false, computes discriminator loss, else generator loss.

        Returns:
            loss: The combined loss.
            log: A dictionary containing all sub-losses and the total loss.
        """
        device = x.device
        log = {}
        loss = torch.tensor([0.0]).to(device)

        # loss for generator / autoencoder
        if train_autoencoder:
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

            if self.discriminator is not None:
                disc_out_fake = self.discriminator(x_hat)
                target_fake = torch.ones(disc_out_fake.shape).to(device)
                generator_loss = nn.functional.binary_cross_entropy(disc_out_fake, target_fake)
                generator_loss *= self.disc_weight
                loss += generator_loss
                log['generator_loss'] = generator_loss.item()

            log['loss'] = loss.item()
        # loss for discriminator
        else:
            # fake output
            disc_out_fake = self.discriminator(x_hat.detach())
            target_fake = torch.zeros(disc_out_fake.shape).to(device)
            disc_loss_fake = nn.functional.binary_cross_entropy(disc_out_fake, target_fake)
            log['disc_loss_fake'] = disc_loss_fake.item()

            # real output
            disc_out_real = self.discriminator(x.detach())
            target_real = torch.ones(disc_out_real.shape).to(device)
            disc_loss_real = nn.functional.binary_cross_entropy(disc_out_real, target_real)
            log['disc_loss_real'] = disc_loss_real.item()

            disc_loss = 0.5 * (disc_loss_real + disc_loss_fake)
            disc_loss *= self.disc_weight
            log['disc_loss'] = disc_loss.item()
            loss += disc_loss

        return loss, log


if __name__ == "__main__":
    l_vqgan = LossFn(perceptual_weight=1)

    ipt = torch.randn((8, 3, 128, 128))

    latent_e = torch.randn((8, 10, 32, 32))
    latent_q = torch.randn((8, 10, 32, 32))

    out, logs = l_vqgan(ipt, ipt, latent_e, latent_q)       # train autoencoder
    print("Output:", out)
    print("Logs:", logs)

    out, logs = l_vqgan(ipt, ipt, latent_e, latent_q, train_autoencoder=False)       # train discriminator
    print("Output:", out)
    print("Logs:", logs)
