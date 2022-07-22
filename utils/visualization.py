import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image


def normalize(img):
    img *= 127.5
    img += 127.5

    return img


def tensor_to_image(tensor):
    """
    Convert torch.Tensor to PIL image.
    """
    n_channels = tensor.shape[0]

    img = tensor.detach().cpu()
    img = normalize(img)

    if n_channels > 1:
        return Image.fromarray(img.permute(1, 2, 0).numpy().astype(np.uint8)).convert("RGB")
    else:
        return Image.fromarray(img[0].numpy()).convert("L")


def get_reconstruction_figure(x: torch.Tensor, x_hat: torch.Tensor, n_ims: int = 8,
                              fig_size: tuple = (14, 4), dpi: int = 150):
    """
    Returns figure of original and reconstruction images. Top row are originals, bottom
    row are reconstructions.

    Args:
        x: Original image tensor
        x_hat: Reconstructed image tensor
        n_ims: Number of images of that batch to be plotted
        fig_size: Size of the figure
        dpi: Resolution

    Returns:
        fig: Matplotlib figure
    """
    bs, c, h, w = x.shape

    n_ims = n_ims if n_ims <= bs else bs
    cmap = None if c > 1 else 'gray'

    fig, axes = plt.subplots(nrows=2, ncols=n_ims, figsize=fig_size,
                             dpi=dpi, tight_layout=True, squeeze=True,
                             gridspec_kw={'wspace': 0, 'hspace': 0})

    for im_idx in range(n_ims):
        # original
        im = tensor_to_image(x[im_idx])

        ax = axes[0, im_idx]
        ax.imshow(im, cmap=cmap)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        if im_idx == 0:
            ax.set_ylabel("Original")

        # reconstruction
        im = tensor_to_image(x_hat[im_idx])

        ax = axes[1, im_idx]
        ax.imshow(im, cmap=cmap)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        if im_idx == 0:
            ax.set_ylabel("Reconstruction")

    return fig
