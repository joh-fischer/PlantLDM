from typing import Dict, Tuple

import torch
import torch.nn as nn

blk = lambda ic, oc: nn.Sequential(
    nn.Conv2d(ic, oc, 7, padding=3),
    nn.BatchNorm2d(oc),
    nn.LeakyReLU(),
)


class DummyEpsModel(nn.Module):
    """
    This should be unet-like, but let's don't think about the model too much :P
    Basically, any universal R^n -> R^n model should work.
    """

    def __init__(self, n_channel: int) -> None:
        super(DummyEpsModel, self).__init__()
        self.conv = nn.Sequential(  # with batchnorm
            blk(n_channel, 64),
            blk(64, 128),
            blk(128, 256),
            blk(256, 512),
            blk(512, 256),
            blk(256, 128),
            blk(128, 64),
            nn.Conv2d(64, n_channel, 3, padding=1),
        )

        self.tanh = nn.Tanh()

    def forward(self, x, t) -> torch.Tensor:
        # Lets think about using t later. In the paper, they used Tr-like positional embeddings.
        x = self.conv(x)
        # normalize to [-1, 1]
        x = self.tanh(x)
        return x


class DDPM(nn.Module):
    def __init__(
            self,
            eps_model: nn.Module,
            betas: Tuple[float, float],
            n_steps: int,
            img_channels: int = 3,
            img_size: Tuple[int, int] = (128, 128),
            criterion: nn.Module = nn.MSELoss(),
    ):
        super(DDPM, self).__init__()

        assert betas[0] < betas[1] < 1.0, "beta1 < beta2 < 1.0 not fulfilled"

        self.eps_model = eps_model
        self.n_steps = n_steps
        self.img_channels = img_channels
        self.img_size = img_size
        self.criterion = criterion

        # beta values
        self.betas = betas
        self.beta = self.linear_beta_schedule()
        self.sqrt_beta = torch.sqrt(self.beta)

        # alpha values
        self.alpha = 1.0 - self.beta
        self.log_alpha = torch.log(self.alpha)
        self.oneover_sqrta = 1 / torch.sqrt(self.alpha)
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sqrtab = torch.sqrt(self.alpha_bar)
        self.sqrtmab = torch.sqrt(1 - self.alpha_bar)
        self.mab_over_sqrtmab_inv = (1 - self.alpha) / self.sqrtmab

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """
        t = torch.randint(1, self.n_steps, (x.shape[0],)).to(x.device)  # t ~ Uniform({1, ..., T})

        eps = torch.randn_like(x)  # epsilon ~ N(0, 1)

        # TODO: put all of the variables on the x device
        x_t = (self.sqrtab.to(x.device)[t, None, None, None] * x
               + self.sqrtmab.to(x.device)[t, None, None, None] * eps
               )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        return self.criterion(eps, self.eps_model(x_t, t / self.n_steps))

    @torch.no_grad()
    def sample(self, batch_size: int, device) -> torch.Tensor:
        x = torch.randn(batch_size, self.img_channels, *self.img_size).to(device)  # x_T ~ N(0, 1)

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for t in range(self.n_steps - 1, -1, -1):
            z = torch.randn(batch_size, self.img_channels, *self.img_size).to(device) if t > 0 else 0
            eps = self.eps_model(x, t / self.n_steps)
            x = (
                    self.oneover_sqrta[t]
                    * (x - eps * self.mab_over_sqrtmab_inv[t])
                    + self.sqrt_beta[t] * z
            )

        return x

    def cosine_beta_schedule(self, s: float = 0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = self.n_steps + 1
        x = torch.linspace(0, self.n_steps, steps)
        alphas_cumprod = torch.cos(((x / self.n_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        beta_values = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(beta_values, 0.0001, 0.9999)

    def linear_beta_schedule(self):
        return torch.linspace(self.betas[0], self.betas[1], self.n_steps)

    def quadratic_beta_schedule(self):
        return torch.linspace(self.betas[0] ** 0.5, self.betas[1] ** 0.5, self.n_steps) ** 2

    def sigmoid_beta_schedule(self):
        beta_values = torch.linspace(-6, 6, self.n_steps)
        return torch.sigmoid(beta_values) * (self.betas[1] - self.betas[0]) + self.betas[0]
