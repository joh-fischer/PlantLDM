import os
import sys
import time
import yaml
import argparse
from tqdm import tqdm
from datetime import datetime
import torch

from model.vqvae import VQVAE
from utils.logger import Logger
from utils.helpers import timer
from utils.helpers import load_vqvae_checkpoint, save_vqvae_checkpoint
from utils.visualization import get_original_reconstruction_figure
from dataloader import CIFAR10, PlantNet


CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'logs'
TIMESTAMP = datetime.now().strftime('%y-%m-%d_%H%M%S')


parser = argparse.ArgumentParser(description="PyTorch First Stage Training")
parser.add_argument('--name', '-n', default='',
                    type=str, metavar='NAME', help='Model name and folder where logs are stored')
parser.add_argument('--epochs', default=2,
                    type=int, metavar='N', help='Number of epochs to run (default: 2)')
parser.add_argument('--batch-size', default=64, metavar='N',
                    type=int, help='Mini-batch size (default: 64)')
parser.add_argument('--lr', default=0.0001,
                    type=float, metavar='LR', help='Initial learning rate (default: 0.0001)')
parser.add_argument('--config', default='configs/vqvae.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/vqvae.yaml)')
parser.add_argument('--data-config', default='configs/data_jo.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/data_jo.yaml)')
parser.add_argument('--debug', action='store_true',
                    help='If true, trains on CIFAR10')
parser.add_argument('--gpus', default=0, type=int,
                    nargs='+', metavar='GPUS', help='If GPU(s) available, which GPU(s) to use for training.')
parser.add_argument('--ckpt-save', default=True, action=argparse.BooleanOptionalAction,
                    dest='save_checkpoint', help='Save checkpoints to folder')
parser.add_argument('--load-ckpt', default=None, metavar='PATH',
                    dest='load_checkpoint', help='Load model checkpoint and continue training')
parser.add_argument('--log-save-interval', default=5, type=int, metavar='N',
                    dest='save_interval', help="Interval in which logs are saved to disk (default: 5)")

logger = Logger(LOG_DIR)


def main():
    args = parser.parse_args()
    for name, val in vars(args).items():
        print("{:<16}: {}".format(name, val))

    # setup paths and logging
    args.name = 'first_stage/' + args.name
    running_log_dir = os.path.join(LOG_DIR, args.name, f'{TIMESTAMP}')
    running_ckpt_dir = os.path.join(CHECKPOINT_DIR, args.name, f'{TIMESTAMP}')
    print("{:<16}: {}".format('logdir', running_log_dir))
    print("{:<16}: {}".format('ckpt_dir', running_ckpt_dir))

    if args.save_checkpoint and not os.path.exists(running_ckpt_dir):
        os.makedirs(running_ckpt_dir)

    global logger
    logger = Logger(running_log_dir, tensorboard=True)

    # setup GPU
    args.gpus = args.gpus if isinstance(args.gpus, list) else [args.gpus]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in args.gpus])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("{:<16}: {}".format('device', device))
    if torch.cuda.device_count() > 1:
        raise ValueError('Currently multi-gpu training is not possible')

    # load data
    if args.debug:
        data = CIFAR10(args.batch_size)
    else:
        data_cfg = yaml.load(open(args.data_config, 'r'), Loader=yaml.Loader)
        data = PlantNet(**data_cfg, batch_size=args.batch_size)

    # read config file for model
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # create model and optimizer
    model = VQVAE(**cfg)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # resume training
    if args.load_checkpoint:
        model, start_epoch = load_vqvae_checkpoint(model, args.load_checkpoint, device)
        args.epochs += start_epoch
    else:
        start_epoch = 0

    # start run
    logger.log_hparams({**cfg, **vars(args)})
    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        logger.init_epoch(epoch)
        print(f"Epoch [{epoch + 1} / {args.epochs}]")

        train(model, data.train, optimizer, device)

        validate(model, data.val, device)

        # logging
        output = [f'{k}: {v.avg:.4f}' for k, v in logger.epoch.items()]
        print(*output, sep=' - ')
        for k, v in logger.epoch.items():
            logger.tensorboard.add_scalar(k, v.avg, global_step=epoch)

        # save logs and checkpoint
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            logger.save()
            if args.save_checkpoint:
                save_vqvae_checkpoint(model, running_ckpt_dir, logger)

    elapsed_time = timer(t_start, time.time())
    print(f"Total training time: {elapsed_time}")


def train(model, train_loader, optimizer, device):
    model.train()

    is_first = True
    for x, _ in tqdm(train_loader, desc="Training"):
        x = x.to(device)

        x_hat, emb_loss = model(x)

        rec_loss = torch.nn.functional.mse_loss(x_hat, x)

        loss = rec_loss + emb_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics = {'rec_loss': rec_loss, 'emb_loss': emb_loss, 'loss': loss}
        logger.log_metrics(metrics, phase='train', aggregate=True, n=x.shape[0])

        if is_first:
            is_first = False
            logger.tensorboard.add_figure('Train: Original vs. Reconstruction',
                                          get_original_reconstruction_figure(x, x_hat, n_ims=8),
                                          global_step=logger.running_epoch)


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()

    is_first = True
    for x, _ in tqdm(val_loader, desc="Validation"):
        x = x.to(device)

        x_hat, emb_loss = model(x)

        rec_loss = torch.nn.functional.mse_loss(x_hat, x)

        loss = rec_loss + emb_loss

        metrics = {'val_rec_loss': rec_loss, 'val_emb_loss': emb_loss, 'val_loss': loss}
        logger.log_metrics(metrics, phase='val', aggregate=True, n=x.shape[0])

        if is_first:
            is_first = False
            logger.tensorboard.add_figure('Val: Original vs. Reconstruction',
                                          get_original_reconstruction_figure(x, x_hat, n_ims=8),
                                          global_step=logger.running_epoch)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exit training with keyboard interrupt!")
        logger.save()
        sys.exit(0)