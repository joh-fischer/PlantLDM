import torch
import torch.nn as nn

from model.layers import ResidualBlock


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3, n_layers: int = 2,
                 start_channels: int = 32, residual_blocks: bool = False):
        """
        Discriminator for VQ-GAN, with option to add residual blocks to increase
        model capacity.

        Args:
            in_channels: Input channels (Default: 3).
            n_layers: Number of down-sampling layers. Final resolution will be
                image size divided by 2 ** n_layers.
            start_channels: Number of starting channels, which get multiplied up
                until 2 ** n_layers.
            residual_blocks: If True, adds residual blocks between the down-sampling
                layers.
        """
        super().__init__()

        k = 4
        s = 2
        p = 1

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, start_channels, kernel_size=k, stride=s, padding=p),
            nn.LeakyReLU(0.2, True)
        )

        prev_channels = start_channels
        self.blocks = nn.ModuleList([])
        for n in range(1, n_layers+1):
            channel_mult = min(2 ** n, 4)
            out_channels = start_channels * channel_mult
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels, out_channels, k, s, p),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, True),
                    ResidualBlock(out_channels, out_channels) if residual_blocks else nn.Identity()
                )
            )
            prev_channels = out_channels

        self.out_conv = nn.Conv2d(prev_channels, 1, k, s, p)
        self.apply(weights_init)

    def forward(self, x: torch.Tensor):
        x = self.init_conv(x)

        for block in self.blocks:
            x = block(x)

        x = self.out_conv(x)
        x = torch.sigmoid(x)

        return x


if __name__ == "__main__":
    disc = Discriminator(n_layers=4, residual_blocks=True)
    print(disc)

    ipt = torch.randn((8, 3, 128, 128))
    out = disc(ipt)

    print("Input:", ipt.shape)      # [bs, 3, 128, 128]
    print("Output:", out.shape)     # [bs, 1, 8, 8]
