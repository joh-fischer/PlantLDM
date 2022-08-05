import numpy as np
import matplotlib.pyplot as plt
import torch
import einops
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


def get_original_reconstruction_image(x: torch.Tensor, x_hat: torch.Tensor, n_ims: int = 8) -> object:
    """
    Returns pillow image of original and reconstruction images. Top row are originals, bottom
    row are reconstructions. Faster than creating a figure.

    Args:
        x: Original image tensor
        x_hat: Reconstructed image tensor
        n_ims: Number of images of that batch to be plotted

    Returns:
        ims: Numpy array in shape [h, w, 3] with top row being originals and
            bottom row being reconstructions.
    """
    bs, c, h, w = x.shape

    n_ims = n_ims if n_ims <= bs else bs

    x = x[:n_ims, ...].detach().cpu().numpy()
    x_hat = x_hat[:n_ims, ...].detach().cpu().numpy()

    x = np.transpose(x, (0, 2, 3, 1))
    x_hat = np.transpose(x_hat, (0, 2, 3, 1))

    x = einops.rearrange(x, 'b h w c -> h (b w) c')
    x_hat = einops.rearrange(x_hat, 'b h w c -> h (b w) c')

    ims = np.concatenate((x, x_hat), axis=0)

    ims = (ims * 127.5 + 127.5).astype(np.uint8)

    return ims


def get_original_reconstruction_figure(x: torch.Tensor, x_hat: torch.Tensor, n_ims: int = 8,
                                       fig_size: tuple = (14, 4), dpi: int = 150):
    """
    Returns figure of original and reconstruction images. Top row are originals, bottom
    row are reconstructions. Slower but larger images.

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


def get_sample_images_for_ddpm(images: list(), n_ims: int = 8, fig_size: tuple = (8, 10), dpi: int = 150):
    """
    Returns figure of original and reconstruction images. Top row are originals, bottom
    row are reconstructions. Slower but larger images.

    Args:
        images: List of sampled images
        n_ims: Number of images that should be plotted
        fig_size: Size of the figure
        dpi: Resolution

    Returns:
        fig: Matplotlib figure
    """
    bs, c, h, w = images[0].shape

    n_cols = 10
    col_idxs = np.linspace(0, len(images) - 1, n_cols, dtype=int)

    n_ims = n_ims if n_ims <= bs else bs
    cmap = None if c > 1 else 'gray'

    fig, axes = plt.subplots(nrows=n_ims, ncols=n_cols, figsize=fig_size,
                             dpi=dpi, gridspec_kw={'wspace': 0, 'hspace': 0})

    for im_idx in range(n_cols):
        for row_idx in range(n_ims):
            im = tensor_to_image(images[col_idxs[im_idx]][row_idx])

            ax = axes[row_idx, im_idx]
            ax.imshow(im, cmap=cmap)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

            if row_idx == 0:
                ax.set_xlabel(f"t = {col_idxs[im_idx]}", fontsize=6)
                ax.xaxis.set_label_position('top')

            if im_idx == 0:
                ax.set_ylabel(f"Image {row_idx + 1}", fontsize=6)

    return fig


if __name__ == "__main__":
    import time

    x_ = torch.randn((16, 3, 128, 128))
    x_hat_ = torch.randn((16, 3, 128, 128))

    t0 = time.time()
    _ = get_original_reconstruction_image(x_, x_hat_)
    print("time:", time.time() - t0)

    t0 = time.time()
    _ = get_original_reconstruction_figure(x_, x_hat_)
    print("time:", time.time() - t0)
