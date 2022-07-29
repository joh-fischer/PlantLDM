from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class DDPM(nn.Module):
    def __init__(
            self,
            eps_model: nn.Module,
            beta_1: float,
            beta_2: float,
            beta_schedule: str,
            n_steps: int,
            loss_function: str,
    ):
        super(DDPM, self).__init__()
        self.n_steps = n_steps
        self.eps_model = eps_model

        if not beta_1 < beta_2 < 1.0:
            raise ValueError(f"beta1: {beta_1} < beta2: {beta_2} < 1.0 not fulfilled")

        available_beta_schedules = ["linear", "quadratic", "sigmoid", "cosine"]
        if beta_schedule not in available_beta_schedules:
            raise ValueError(f"Beta schedule should be one of the following: {available_beta_schedules}")

        available_loss_functions = ["l1", "l2", "huber"]
        if loss_function not in available_loss_functions:
            raise ValueError(f"Loss function should be one of the following: {available_loss_functions}")
        self.loss_function = loss_function

        # define beta schedule
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.betas = None
        if beta_schedule == "linear":
            self.betas = self.linear_beta_schedule()
        elif beta_schedule == "quadratic":
            self.betas = self.quadratic_beta_schedule()
        elif beta_schedule == "sigmoid":
            self.betas = self.sigmoid_beta_schedule()
        elif beta_schedule == "cosine":
            self.betas = self.cosine_beta_schedule()

        # define alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def extract(self, a, t, x_shape):
        """
        extracts an appropriate t index for a batch of indices
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_start, t, noise=None):
        """
        forward diffusion process
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def get_noisy_image(self, x_start, t):
        """
        Gets a noisy image for a certain timestep
        """
        # add noise
        x_noisy = self.q_sample(x_start, t=t)

        return x_noisy

    def p_losses(self, x_start, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        t = torch.randint(0, self.n_steps, (x_start.shape[0],)).to(x_start.device)  # t ~ Uniform({1, ..., T})

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.eps_model(x_noisy, t)

        if self.loss_function == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        elif self.loss_function == "l2":
            loss = F.mse_loss(noise, predicted_noise)
        elif self.loss_function == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in https://arxiv.org/abs/2006.11239
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * self.eps_model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        """
        Implements Algorithm 2 of https://arxiv.org/abs/2006.11239
        """
        device = next(self.eps_model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(reversed(range(0, self.n_steps)), desc='sampling loop time step', total=self.n_steps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img)
        return imgs

    @torch.no_grad()
    def sample(self, image_size, batch_size=16, channels=3):
        return self.p_sample_loop(shape=(batch_size, channels, image_size, image_size))

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
        return torch.linspace(self.beta_1, self.beta_2, self.n_steps)

    def quadratic_beta_schedule(self):
        return torch.linspace(self.beta_1 ** 0.5, self.beta_2 ** 0.5, self.n_steps) ** 2

    def sigmoid_beta_schedule(self):
        beta_values = torch.linspace(-6, 6, self.n_steps)
        return torch.sigmoid(beta_values) * (self.beta_2 - self.beta_1) + self.beta_1
