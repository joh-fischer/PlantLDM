import argparse
import os
import pathlib
import time
from datetime import datetime

import torch
import yaml

from tqdm import tqdm

from dataloader import PlantNet, CIFAR10
from model import VQGANLight
from model.ddpm.ddpm import DDPM
from model.unet import UNet
from model.unet.unet_light import UNetLight
from utils.helpers import timer, save_model_checkpoint, load_model_checkpoint, log2tensorboard_ddpm
from utils.logger import Logger
from utils.visualization import get_sample_images_for_ddpm

# TODO: check if this is necessary
# from: https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-initializing-libiomp5md-dll-but-found-mk2iomp5md-dll-a
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CHECKPOINT_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), 'checkpoints')
LOG_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), 'logs')
TIMESTAMP = datetime.now().strftime('%y-%m-%d_%H%M%S')

parser = argparse.ArgumentParser(description="PyTorch Second Stage Training")
parser.add_argument('--name', '-n', default='',
                    type=str, metavar='NAME', help='Model name and folder where logs are stored')
parser.add_argument('--epochs', default=100,
                    type=int, metavar='N', help='Number of epochs to run (default: 100)')
parser.add_argument('--batch-size', default=64, metavar='N',
                    type=int, help='Mini-batch size (default: 64)')
parser.add_argument('--image-size', default=128, metavar='N',
                    type=int, help='Size that images should be resized to before processing (default: 128)')
parser.add_argument('--image-channels', default=3, metavar='N',
                    type=int, help='Number of image channels (default: 3)')
parser.add_argument('--num-workers', default=0, metavar='N',
                    type=int, help='Number of workers for the dataloader (default: 0)')
parser.add_argument('--lr', default=0.0002,
                    type=float, metavar='LR', help='Initial learning rate (default: 0.0002)')
parser.add_argument('--config', default='configs/ddpm.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/ddpm.yaml)')
parser.add_argument('--unet-config', default='configs/unet.yaml',
                    metavar='PATH', help='Path to unet model config file (default: configs/unet.yaml)')
parser.add_argument('--data-config', default='configs/data_se.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/data_se.yaml)')
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
parser.add_argument('--vae-path', default='',
                    metavar='PATH', help='Path to encoder/decoder model checkpoint (default: empty)')
parser.add_argument('--vae-config', default='configs/vqgan.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/vqgan.yaml)')

logger = Logger(LOG_DIR)


def main():
    args = parser.parse_args()
    for name, val in vars(args).items():
        print("{:<16}: {}".format(name, val))

    # setup paths and logging
    args.name = 'second_stage/' + args.name
    running_log_dir = os.path.join(LOG_DIR, args.name, f'{TIMESTAMP}')
    running_ckpt_dir_ddpm = os.path.join(CHECKPOINT_DIR, args.name, f'{TIMESTAMP}', "ddpm")
    running_ckpt_dir_unet = os.path.join(CHECKPOINT_DIR, args.name, f'{TIMESTAMP}', "unet")
    print("{:<16}: {}".format('logdir', running_log_dir))
    print("{:<16}: {}".format('ckpt_dir_ddpm', running_ckpt_dir_ddpm))
    print("{:<16}: {}".format('ckpt_dir_unet', running_ckpt_dir_unet))

    if args.save_checkpoint and not os.path.exists(running_ckpt_dir_ddpm):
        os.makedirs(running_ckpt_dir_ddpm)

    if args.save_checkpoint and not os.path.exists(running_ckpt_dir_unet):
        os.makedirs(running_ckpt_dir_unet)

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
        data = PlantNet(**data_cfg, batch_size=args.batch_size, image_size=args.image_size,
                        num_workers=args.num_workers)

    # read config file for model
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfg_unet = yaml.load(open(args.unet_config, 'r'), Loader=yaml.Loader)
    cfg_vae = yaml.load(open(args.vae_config, 'r'), Loader=yaml.Loader)

    vae_model = VQGANLight(**cfg_vae['model'])
    vae_model, _, _ = load_model_checkpoint(vae_model, args.vae_path, device)

    unet = UNetLight(**cfg_unet)
    unet.to(device)

    ddpm = DDPM(eps_model=unet, vae_model=vae_model, **cfg)
    ddpm.to(device)

    optimizer = torch.optim.Adam(unet.parameters(), args.lr)

    # resume training
    if args.load_checkpoint:
        ddpm, start_epoch, global_train_step = load_model_checkpoint(ddpm, args.load_checkpoint, device)
        logger.global_train_step = global_train_step
        args.epochs += start_epoch
    else:
        start_epoch = 0

    # start run
    logger.log_hparams({**cfg, **vars(args)})
    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):

        logger.init_epoch(epoch)
        logger.global_train_step = logger.running_epoch
        print(f"Epoch [{epoch + 1} / {args.epochs}]")

        train(ddpm, data.train, optimizer, device)

        validate(ddpm)

        # logging
        output = ' - '.join([f'{k}: {v.avg:.4f}' for k, v in logger.epoch.items()])
        print(output)

        # save logs and checkpoint
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            logger.save()
            if args.save_checkpoint:
                save_model_checkpoint(unet, f"{running_ckpt_dir_ddpm}", logger)
                save_model_checkpoint(ddpm, f"{running_ckpt_dir_unet}", logger)

    elapsed_time = timer(t_start, time.time())
    print(f"Total training time: {elapsed_time}")


def train(model, train_loader, optimizer, device):
    model.train()

    ema_loss = None
    for x, _ in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        x = x.to(device)
        loss = model.p_losses(x)
        loss.backward()
        optimizer.step()

        if ema_loss is None:
            ema_loss = loss.item()
        else:
            ema_loss = 0.9 * ema_loss + 0.1 * loss.item()

        metrics = {'ema_loss': ema_loss, 'loss': loss}
        logger.log_metrics(metrics, phase='train', aggregate=True, n=x.shape[0])

        if logger.global_train_step % 150 == 0:
            log2tensorboard_ddpm(logger, 'Train DDPM', ['ema_loss', 'loss'])


@torch.no_grad()
def validate(model):
    model.eval()

    n_images = 8
    images = model.sample(32, batch_size=n_images, channels=3)

    logger.tensorboard.add_figure('Val: DDPM',
                                  get_sample_images_for_ddpm(images, n_ims=n_images),
                                  global_step=logger.global_train_step)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.save()
        raise e
