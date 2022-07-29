import torch
import torch.nn as nn

from model.layers.quantizer import VectorQuantizer
from model.vqgan.layers.encoder import Encoder
from model.vqgan.layers.decoder import Decoder


class VQGAN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        pass
