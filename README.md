# PlantLDM

## Planned Timeline:
20.07.
- [X] VQVAE

27.07.
- [X] Training Loop Implementation VQ-VAE
- [ ] VQGAN Implementation
- [ ] U-Net Implementation
- [ ] DDPM
- [ ] Training Loop Implementation DDPM
- [ ] Training of VQGAN

29.07.
- [ ] Deployment (somewhere)
- [ ] Full training with CIFAR10
- [ ] Deployment of resized and cropped plant images (128x128)

01.08.
- [ ] Adjustments to models 

04.08.
- [ ] Training with PlantNet


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
python3 train_first_stage.py --name run/e10 --epochs 10 --data-config config/data_jo.yaml
```

To first debug the code with `CIFAR10` just run

```
python3 train_first_stage.py --name debug --epochs 10 --debug
```


### Help message

You can print the help message with `python3 train_first_stage.py -h`.

```
PyTorch First Stage Training

options:
  -h, --help            show this help message and exit
  --name NAME, -n NAME  Model name and folder where logs are stored
  --epochs N            Number of epochs to run (default: 2)
  --batch-size N        Mini-batch size (default: 64)
  --lr LR               Initial learning rate (default: 0.0001)
  --config PATH         Path to model config file (default: configs/vqvae.yaml)
  --data-config PATH    Path to model config file (default: configs/data_johannes.yaml)
  --debug               If true, trains on CIFAR10
  --gpus GPUS [GPUS ...]
                        If GPU(s) available, which GPU(s) to use for training.
  --ckpt-save, --no-ckpt-save
                        Save checkpoints to folder (default: True)
  --load-ckpt PATH      Load model checkpoint and continue training
  -e, --evaluate        Evaluate model on test set
  --log-save-interval N
                        Interval in which logs are saved to disk (default: 5)
```
