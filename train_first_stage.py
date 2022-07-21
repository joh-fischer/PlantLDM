import os
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
from test_data.dataloader import CIFAR10        # TODO: real data


CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'logs'
TIMESTAMP = datetime.now().strftime('%y-%m-%d_%H%M%S')


parser = argparse.ArgumentParser(description="PyTorch First Stage Training")
parser.add_argument('--name', '-n', default='',
                    type=str, metavar='NAME', help='Model name and folder where logs are stored')
parser.add_argument('--epochs', default=2,
                    type=int, metavar='N', help='Number of epochs to run (default: 2)')
parser.add_argument('--batch-size', default=16, metavar='N',
                    type=int, help='Mini-batch size (default: 16)')
parser.add_argument('--lr', default=0.001,
                    type=float, metavar='LR', help='Initial learning rate (default: 0.001)')
parser.add_argument('--config', default='configs/vqvae.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/vqvae.yaml)')
parser.add_argument('--gpus', default=0, type=int,
                    nargs='+', metavar='GPUS', help='If GPU(s) available, which GPU(s) to use for training.')
parser.add_argument('--ckpt-save', default=True, action=argparse.BooleanOptionalAction,
                    dest='save_checkpoint', help='Save checkpoints to folder')
parser.add_argument('--load-ckpt', default=None, metavar='PATH',
                    dest='load_checkpoint', help='Load model checkpoint and continue training')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='Evaluate model on test set')
parser.add_argument('--log-save-interval', default=5, type=int, metavar='N',
                    dest='save_interval', help="Interval in which logs are saved to disk (default: 5)")

logger = Logger(LOG_DIR)


def main():
    args = parser.parse_args()
    for name, val in vars(args).items():
        print("{:<16}: {}".format(name, val))

    # setup paths and logging
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
    data = CIFAR10(args.batch_size)

    # read config file
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # create model and optimizer
    model = VQVAE(**cfg)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # resume training
    if args.load_checkpoint:
        model, optimizer, start_epoch = load_vqvae_checkpoint(model, optimizer, args.load_checkpoint, device)
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

        print("Loss: ", logger.epoch['loss'].avg)

        # save logs and checkpoint
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            logger.save()
            if args.save_checkpoint:
                save_vqvae_checkpoint(model, optimizer, running_ckpt_dir, logger)

    elapsed_time = timer(t_start, time.time())
    print(f"Total training time: {elapsed_time}")


def train(model, train_loader, optimizer, device):
    model.train()
    for x, _ in tqdm(train_loader, desc="Training"):
        x = x.to(device)

        z_e, z_q, z_q_sg, x_hat = model(x)

        # log p(x | z_q(x))
        recon_loss = torch.nn.functional.mse_loss(x_hat, x)
        # ( sg[z_e(x)] - e )
        embedding_loss = torch.nn.functional.mse_loss(z_q, z_e.detach())
        # beta * ( z_e(x) - sg[e] )
        commitment_loss = torch.nn.functional.mse_loss(z_q.detach(), z_e)

        loss = recon_loss + embedding_loss + 0.25 * commitment_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.log_metrics({'loss': loss.item()}, phase='train', aggregate=True, n=x.shape[0])


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    pass


if __name__ == "__main__":
    main()
