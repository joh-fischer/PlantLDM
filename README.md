# PlantLDM

## TODO's

- [ ] Make loss config for vqgan singleton, s.t. we can easily change it
- [ ] Test discriminator (including warm-up phase)
- [ ] conduct experiments with VQ-GAN
- [ ] Implement training script of DDPM with encoder and decoder of first stage training
- [ ] Yu et al (2022) state that codebook is merely used in VQ-GAN. Maybe apply some proposals of them to enhance it?

## Experiments

### VQ-GAN
- train only with L2 and codebook loss (VQ-VAE with better encoder-decoder)
  - per epoch approx. 13 min
- train only with perceptual loss, no adversarial loss
  - per epoch approx. 23 min
- train only with adversarial loss, no perceptual loss
- vary warm up iterations


### Compare L1 to L2

- run VQVAE with L1 and L2, respectively
- compare results
- is Isola paper right, less blurry results with L1?

### Embedding Loss
- embedding loss first increases, then drops, and then increases again
- can we tackle that problem with more epochs?


### Change Loss Parameters
- Yu et al. ((2022)[https://arxiv.org/abs/2110.04627]) used different parameters
- $L = L_{VQ} + 0.1 L_{Adv} + 0.1 L_{Perceptual} + 0.1 L_{Logit-Laplace} + 1.0 L_{2}$

## Setup

Clone the repository and create a virtual environment (optional)

```
virtualenv venv
source venv/bin/activate
```

Then install the dependencies
```
pip install -r requirements.txt
```

## Tensorboard

In order to monitor the losses and visualizations just `cd` into the repo and run
```
tensorboard --logdir=logs
```

## VQ-VAE

Implementation of VQ-VAE ([paper](https://arxiv.org/abs/1711.00937v2)).

### Usage

For the usage on the dataset specified in the data config (`configs/data_<>.yaml`) you can run the following on the
command line

```
python3 train_vqvae.py --name run/e10 --epochs 10 --data-config configs/data_jo.yaml
```

To first debug the code with `CIFAR10` just run

```
python3 train_vqvae.py --name mydebug --epochs 10 --debug
```


### Usage

You can print the help message with `python3 train_first_stage.py -h`.

```
usage: train_vqvae.py [-h] [--name NAME] [--epochs N] [--batch-size N] [--image-size N]
                      [--num-workers N] [--lr LR] [--config PATH] [--data-config PATH] [--debug]
                      [--gpus GPUS [GPUS ...]] [--ckpt-save | --no-ckpt-save] [--load-ckpt PATH]
                      [--log-save-interval N]

PyTorch First Stage Training

optional arguments:
  -h, --help            show this help message and exit
  --name NAME, -n NAME  Model name and folder where logs are stored
  --epochs N            Number of epochs to run (default: 2)
  --batch-size N        Mini-batch size (default: 64)
  --image-size N        Size that images should be resized to before processing (default: 128)
  --num-workers N       Number of workers for the dataloader (default: 0)
  --lr LR               Initial learning rate (default: 0.0001)
  --config PATH         Path to model config file (default: configs/vqvae.yaml)
  --data-config PATH    Path to model config file (default: configs/data_se.yaml)
  --debug               If true, trains on CIFAR10
  --gpus GPUS [GPUS ...]
                        If GPU(s) available, which GPU(s) to use for training.
  --ckpt-save, --no-ckpt-save
                        Save checkpoints to folder (default: True)
  --load-ckpt PATH      Load model checkpoint and continue training
  --log-save-interval N
                        Interval in which logs are saved to disk (default: 5)
```
