import argparse
import os
import pathlib
import time
from datetime import datetime

import torch
import yaml
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from dataloader import PlantNet, CIFAR10
from model.ddpm import DDPM, DummyEpsModel
from utils.helpers import timer, save_model_checkpoint, load_model_checkpoint, log2tensorboard_ddpm
from utils.logger import Logger
from utils.visualization import get_sample_images

# TODO: check if this is necessary
# from: https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-initializing-libiomp5md-dll-but-found-mk2iomp5md-dll-a
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

CHECKPOINT_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), 'checkpoints')
LOG_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), 'logs')
TIMESTAMP = datetime.now().strftime('%y-%m-%d_%H%M%S')

# TODO: check if all of these are needed (just copied it from vqvae)
parser = argparse.ArgumentParser(description="PyTorch Second Stage Training")
parser.add_argument('--name', '-n', default='',
                    type=str, metavar='NAME', help='Model name and folder where logs are stored')
parser.add_argument('--epochs', default=100,
                    type=int, metavar='N', help='Number of epochs to run (default: 100)')
parser.add_argument('--batch-size', default=64, metavar='N',
                    type=int, help='Mini-batch size (default: 64)')
parser.add_argument('--image-size', default=128, metavar='N',
                    type=int, help='Size that images should be resized to before processing (default: 128)')
parser.add_argument('--num-workers', default=0, metavar='N',
                    type=int, help='Number of workers for the dataloader (default: 0)')
parser.add_argument('--lr', default=0.0002,
                    type=float, metavar='LR', help='Initial learning rate (default: 0.0002)')
parser.add_argument('--config', default='configs/vqvae.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/vqvae.yaml)')
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

logger = Logger(LOG_DIR)


def main():
    args = parser.parse_args()
    for name, val in vars(args).items():
        print("{:<16}: {}".format(name, val))

    # setup paths and logging
    args.name = 'second_stage/' + args.name
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
        data = PlantNet(**data_cfg, batch_size=args.batch_size, image_size=args.image_size,
                        num_workers=args.num_workers)

    # read config file for model
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    u_net = DummyEpsModel(3)
    ddpm = DDPM(eps_model=u_net,
                betas=(1e-4, 0.02),
                img_channels=3,
                img_size=(args.image_size, args.image_size),
                n_steps=1000)  # TODO make this an args variable
    ddpm.to(device)

    optimizer = torch.optim.Adam(ddpm.parameters(), args.lr)

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
        print(f"Epoch [{epoch + 1} / {args.epochs}]")

        train(ddpm, data.train, optimizer, device)

        validate(ddpm, data.val, device)

        # logging
        output = ' - '.join([f'{k}: {v.avg:.4f}' for k, v in logger.epoch.items()])
        print(output)

        # save logs and checkpoint
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            logger.save()
            if args.save_checkpoint:
                save_model_checkpoint(ddpm, running_ckpt_dir, logger)

    logger.global_train_step += 1

    elapsed_time = timer(t_start, time.time())
    print(f"Total training time: {elapsed_time}")


def train(model, train_loader, optimizer, device):
    model.train()

    ema_loss = None
    for x, _ in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        x = x.to(device)
        loss = model(x)
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
def validate(model, val_loader, device):
    model.eval()
    # TODO: implement

    # TODO: improve this
    # TODO: make image size an argument
    n_images = 8
    xh = model.sample(n_images, device)
    #grid = make_grid(xh, nrow=4)

    logger.tensorboard.add_figure('Val DDPM',
                                  get_sample_images(xh, n_ims=n_images),
                                  global_step=logger.global_train_step)

    # save model
    # torch.save(model.state_dict(), f"{pathlib.Path(__file__).parent.resolve()}/ddpm_test.pth")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.save()
        raise e
