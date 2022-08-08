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
                 disc_warm_up_iters: int = 500,
                 disc_res_blocks: bool = False,
                 last_decoder_layer: nn.Module = None):
        """
        A class for computing and combining the different losses used in VQ-VAE
        and VQ-GAN.

        Args:
            rec_loss_type: Loss-type for reconstruction loss, either L1 or L2.
            perceptual_weight: Weight for the perceptual LPIPS loss.
            codebook_weight: Weight for the codebook loss of the vector quantizer.
            commitment_weight: Beta for the commitment loss of the vector quantizer.
            disc_weight: Weight for the adversarial loss.
            disc_in_channels: Input channels for the discriminator.
            disc_n_layers: Number of layers in the discriminator.
            disc_res_blocks: If true, adds residual blocks to the discriminator.
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
        # In https://arxiv.org/abs/2012.09841 they set the factor for the adversarial loss
        # to zero for the first iterations (suggestion: at least one epoch). Longer warm-ups
        # generally lead to better reconstructions.
        self.disc_warm_up_iters = disc_warm_up_iters if disc_weight > 0 else None
        self.discriminator = Discriminator(disc_in_channels, n_layers=disc_n_layers,
                                           residual_blocks=disc_res_blocks
                                           ) if disc_weight > 0 else None
        self.opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002
                                         ) if disc_weight > 0 else None
        self.last_layer = last_decoder_layer

    def calculate_adaptive_weight(self, rec_loss, generator_loss):

        rec_grads = torch.autograd.grad(rec_loss, self.last_layer.weight,
                                        retain_graph=True)[0]
        generator_grads = torch.autograd.grad(generator_loss, self.last_layer.weight,
                                              retain_graph=True)[0]

        d_weight = torch.norm(rec_grads) / (torch.norm(generator_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()

        return d_weight

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor,
                z_e: torch.Tensor, z_q: torch.Tensor,
                disc_training: bool = False):
        """
        Computes the final loss including the following sub-losses:
        - reconstruction loss
        - codebook loss for vector quantizer
        - perceptual loss using LPIPS
        - adversarial loss with discriminator

        Args:
            x_hat: Reconstructed image.
            x: Original image.
            z_e: Encoded image.
            z_q: Quantized encoded image.
            disc_training: If true, also trains the discriminator.

        Returns:
            loss: The combined loss.
            log: A dictionary containing all sub-losses and the total loss.
        """
        device = x.device
        log = {}
        loss = torch.tensor([0.0]).to(device)

        # loss for generator / autoencoder
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

        if self.disc_weight > 0 and disc_training:
            disc_out_fake = self.discriminator(x_hat)
            target_fake = torch.ones(disc_out_fake.shape).to(device)
            generator_loss = nn.functional.binary_cross_entropy(disc_out_fake, target_fake)
            generator_loss *= self.disc_weight

            if self.last_layer is not None:
                d_weight = self.calculate_adaptive_weight(rec_loss, generator_loss)
                generator_loss *= d_weight

            loss += generator_loss
            log['generator_loss'] = generator_loss.item()

        log['loss'] = loss.item()

        return loss, log

    def update_discriminator(self, x_hat: torch.Tensor, x: torch.Tensor):
        """
        Updates the discriminator based on the original images and reconstructions.

        Args:
            x_hat: Reconstructed images.
            x: Original images.

        Returns:
            loss: The loss of the discriminator for fake (x_hat) and real (x) images.
            log: A dictionary containing all sub-losses and the total loss.
        """
        device = x.device
        log = {}

        self.discriminator.zero_grad()

        """ Train all real batch """
        # Forward pass real batch through D
        disc_out_real = self.discriminator(x)
        target_real = torch.ones(disc_out_real.shape).to(device)

        # Calculate loss on all-real batch
        disc_loss_real = nn.functional.binary_cross_entropy(disc_out_real, target_real)

        # Calculate gradients for D in backward pass
        disc_loss_real.backward()
        log['disc_loss_real'] = disc_loss_real.item()

        """ Train all fake batch """
        # Classify all fake batch with D
        disc_out_fake = self.discriminator(x_hat.detach())
        target_fake = torch.zeros(disc_out_fake.shape).to(device)

        # Calculate D's loss on the all-fake batch
        disc_loss_fake = nn.functional.binary_cross_entropy(disc_out_fake, target_fake)

        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        disc_loss_fake.backward()
        log['disc_loss_fake'] = disc_loss_fake.item()

        loss = disc_loss_fake + disc_loss_real
        log['disc_loss'] = loss.item()

        """ Update the discriminator """
        # Update D
        self.opt_disc.step()

        return loss, log


if __name__ == "__main__":
    l_vqgan = LossFn(perceptual_weight=1,
                     disc_warm_up_iters=0)

    ipt = torch.randn((8, 3, 128, 128))
    latent_e = torch.randn((8, 10, 32, 32))
    latent_q = torch.randn((8, 10, 32, 32))

    # loss for autoencoder
    out, logs = l_vqgan(ipt, ipt, latent_e, latent_q, disc_training=True)
    print("Output:", out)
    print("Logs:", logs)

    # update discriminator
    out, logs = l_vqgan.update_discriminator(ipt, ipt)
    print("Output:", out)
    print("Logs:", logs)
