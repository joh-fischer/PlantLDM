import os
import torch


def load_model_checkpoint(model, filepath, device):
    """ Load model checkpoint """
    if os.path.isfile(filepath):
        ckpt = torch.load(filepath, map_location=device)
        start_epoch = ckpt['epoch'] + 1
        global_train_step = ckpt['global_train_step']
        model.load_state_dict(ckpt['model_state_dict'])
        print("{:<18}: {} (epoch: {})".format('Loaded checkpoint', filepath, start_epoch))
    else:
        raise ValueError("Checkpoint path '{}' does not exist!".format(filepath))

    return model, start_epoch, global_train_step


def save_model_checkpoint(model, ckpt_dir, logger, prefix=''):
    epoch = logger.running_epoch
    global_train_step = logger.global_train_step
    state = {'epoch': epoch, 'model_state_dict': model.state_dict(),
             'global_train_step': global_train_step}
    filename = os.path.join(ckpt_dir, f'{prefix}e{epoch + 1}.pt')
    print(f"Save checkpoint to '{filename}'")
    torch.save(state, filename)


def log2tensorboard_vqvae(logger, global_tag, tags):
    for tag in tags:
        tb_tag = global_tag + '/' + tag
        logger.tensorboard.add_scalar(tb_tag, logger.epoch[tag].avg,
                                      global_step=logger.global_train_step)


def log2tensorboard_ddpm(logger, global_tag, tags):
    for tag in tags:
        tb_tag = global_tag + '/' + tag
        logger.tensorboard.add_scalar(tb_tag, logger.epoch[tag].avg,
                                      global_step=logger.global_train_step)


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)

    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
