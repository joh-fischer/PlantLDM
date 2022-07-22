import os
import torch


def load_vqvae_checkpoint(model, filepath, device):
    """ Load model checkpoint """
    if os.path.isfile(filepath):
        ckpt = torch.load(filepath, map_location=device)
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['model_state_dict'])
        print("{:<18}: {} (epoch: {})".format('Loaded checkpoint', filepath, start_epoch))
    else:
        raise ValueError("Checkpoint path '{}' does not exist!".format(filepath))

    return model, start_epoch


def save_vqvae_checkpoint(model, ckpt_dir, logger):
    epoch = logger.running_epoch
    state = {'epoch': epoch, 'model_state_dict': model.state_dict()}
    filename = os.path.join(ckpt_dir, f'e{epoch + 1}.pt')
    print(f"Save checkpoint to '{filename}'")
    torch.save(state, filename)


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)

    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
