import torch.nn as nn


class ResidualLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int):
        """
        Residual layer as specified in https://arxiv.org/abs/1711.00937. The original
        layer order is ReLU, 3x3 conv, ReLU, 1x1 conv (paper p. 5) with all having
        256 hidden units.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            hidden_dim (int): number of hidden units
        """
        super(ResidualLayer, self).__init__()

        self.conv3x3 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3,
                                 padding=1, stride=1, bias=False)
        self.conv1x1 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1,
                                 padding=0, stride=1, bias=False)
        self.activation = nn.ReLU(True)

    def forward(self, x):
        identity = x

        x = self.activation(x)
        x = self.conv3x3(x)

        x = self.activation(x)
        x = self.conv1x1(x)

        x += identity

        return x


if __name__ == "__main__":
    import torch

    ipt = torch.randn((32, 64, 128, 128))
    rl = ResidualLayer(64, 64, 128)

    print("Input shape:", ipt.shape)        # [bs, 32, 64, 128, 128]
    print("Output shape:", rl(ipt).shape)   # [bs, 32, 64, 128, 128]
