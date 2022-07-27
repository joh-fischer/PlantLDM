import torch
import torch.nn as nn


class Swish(nn.Module):
    def __init__(self, beta: float = 1.0):
        """
        Swish activation function.

        Args:
            beta: Scalar factor for multiplying x.
        """
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    betas = [0.1, 1.0, 10.]

    ipt = torch.linspace(-5, 3, 1000)

    _, ax = plt.subplots(figsize=(10, 10))
    for b in betas:
        s = Swish(b)
        out = s(ipt)

        ax.plot(ipt, out, label=f'$\\beta = {b}$')

    ax.set_aspect('equal')
    ax.axhline(y=0, color='k', alpha=0.2)
    ax.axvline(x=0, color='k', alpha=0.2)

    plt.legend()
    plt.show()
