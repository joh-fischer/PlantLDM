import torch
from utils.helpers import count_parameters
from model import VQGAN, VQVAE, VQGANLight
from model import UNet, UNetLight
from model.losses import LossFn
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


""" U-Net """
cfg_file = 'configs/unet.yaml'

cfg = yaml.load(open(cfg_file, 'r'), Loader=yaml.Loader)
model = UNetLight(**cfg)

print("U-Net")
print("\t{:<12}: {}".format('total', count_parameters(model)))
print("\t{:<12}: {}".format('down-blocks', count_parameters(model.down_blocks)))
print("\t{:<12}: {}".format('up-blocks', count_parameters(model.up_blocks)))

""" Loss """
cfg_file = 'configs/vqgan.yaml'

cfg = yaml.load(open(cfg_file, 'r'), Loader=yaml.Loader)
model = LossFn(**cfg['loss'])

print("Loss function")
print("\t{:<12}: {}".format('total', count_parameters(model)))
print("\t{:<12}: {}".format('discriminator', count_parameters(model.discriminator)))
print("\t{:<12}: {}".format('perceptual', count_parameters(model.perceptual_loss_fn)))


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

U-Net
    total       : 14273123
    down-blocks : 4197184
    up-blocks   : 7355456
"""
