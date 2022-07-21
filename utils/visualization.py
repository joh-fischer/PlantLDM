import matplotlib.pyplot as plt
import numpy as np


def visualize_reconstruction(x, x_hat, n_images=8):
    bs, c, h, w = x.shape

    n_images = n_images if bs > n_images else bs

