import torch
from utils.helpers import count_parameters
from model import VQGAN, VQVAE, VQGANLight
import yaml


""" VQ-VAE """
cfg_file = 'configs/vqvae.yaml'

cfg = yaml.load(open(cfg_file, 'r'), Loader=yaml.Loader)
model = VQVAE(**cfg['model'])

print("VQ-VAE")
print("\t{:<12}: {}".format('total', count_parameters(model)))
print("\t{:<12}: {}".format('encoder', count_parameters(model.encoder)))
print("\t{:<12}: {}".format('quantizer', count_parameters(model.vq)))
print("\t{:<12}: {}".format('decoder', count_parameters(model.decoder)))

""" VQ-GAN """
cfg_file = 'configs/vqgan.yaml'

cfg = yaml.load(open(cfg_file, 'r'), Loader=yaml.Loader)
model = VQGAN(**cfg['model'])

print("VQ-GAN")
print("\t{:<12}: {}".format('total', count_parameters(model)))
print("\t{:<12}: {}".format('encoder', count_parameters(model.encoder)))
print("\t{:<12}: {}".format('quantizer', count_parameters(model.vq)))
print("\t{:<12}: {}".format('decoder', count_parameters(model.decoder)))

""" VQ-GAN Light """
cfg_file = 'configs/vqgan.yaml'

cfg = yaml.load(open(cfg_file, 'r'), Loader=yaml.Loader)
model = VQGANLight(**cfg['model'])

print("VQ-GAN Light")
print("\t{:<12}: {}".format('total', count_parameters(model)))
print("\t{:<12}: {}".format('encoder', count_parameters(model.encoder)))
print("\t{:<12}: {}".format('quantizer', count_parameters(model.vq)))
print("\t{:<12}: {}".format('decoder', count_parameters(model.decoder)))


""" Results
VQ-VAE
    total       : 109623
    encoder     : 52255
    quantizer   : 5120
    decoder     : 52248

# settings: [32, 64], n_embeddings=512
VQ-GAN
    total       : 1190829
    encoder     : 538442
    quantizer   : 5120
    decoder     : 647267
    
VQ-GAN Light
    total       : 610189
    encoder     : 272618
    quantizer   : 5120
    decoder     : 332451
"""
