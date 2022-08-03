import torch


class BetaSchedule():

    def __init__(
            self,
            beta_1: float,
            beta_2: float,
            beta_schedule: str,
            n_steps: int
    ):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.n_steps = n_steps

        self.values = None
        if beta_schedule == "linear":
            self.values = self._linear_beta_schedule()
        elif beta_schedule == "quadratic":
            self.values = self._quadratic_beta_schedule()
        elif beta_schedule == "sigmoid":
            self.values = self._sigmoid_beta_schedule()
        elif beta_schedule == "cosine":
            self.values = self._cosine_beta_schedule()
        else:
            raise NotImplementedError(f"The requested beta schedule '{beta_schedule}' is not implemented")

    def _cosine_beta_schedule(self, s: float = 0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = self.n_steps + 1
        x = torch.linspace(0, self.n_steps, steps)
        alphas_cumprod = torch.cos(((x / self.n_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        beta_values = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(beta_values, 0.0001, 0.9999)

    def _linear_beta_schedule(self):
        return torch.linspace(self.beta_1, self.beta_2, self.n_steps)

    def _quadratic_beta_schedule(self):
        return torch.linspace(self.beta_1 ** 0.5, self.beta_2 ** 0.5, self.n_steps) ** 2

    def _sigmoid_beta_schedule(self):
        beta_values = torch.linspace(-6, 6, self.n_steps)
        return torch.sigmoid(beta_values) * (self.beta_2 - self.beta_1) + self.beta_1
