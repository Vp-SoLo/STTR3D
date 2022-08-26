import glob
import os
import logging

import torch
from tensorboardX import SummaryWriter


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory
        self.writer = SummaryWriter(log_dir=os.path.join(self.directory))
        print("Start Tensor Board Summary, log_dir = "+directory)

    def config_logger(self, epoch):
        # create logger with 'spam_application'
        logger = logging.getLogger(str(epoch))
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(os.path.join(self.directory, 'epoch_' + str(epoch) + '.log'))
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger


class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('runs', args.dataset, args.checkpoint)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        self.save_experiment_config()

    def save_checkpoint(self, state, filename='model.pth', write_best=True):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)

        best_pred = state['best_pred']
        if write_best:
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))

    def save_experiment_config(self):
        with open(os.path.join(self.experiment_dir, 'parameters.txt'), 'w') as file:
            config_dict = vars(self.args)
            for k in vars(self.args):
                file.write(f"{k}={config_dict[k]} \n")


def save_checkpoint(epoch, model, optimizer, lr_scheduler, prev_best, checkpoint_saver, best, by_iter=False, amp=None):
    """
    Save current state of training
    """

    # save model
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'best_pred': prev_best
    }
    if amp is not None:
        checkpoint['amp'] = amp.state_dict()
    if best:
        checkpoint_saver.save_checkpoint(checkpoint, 'model.pth.tar', write_best=False)
    if by_iter:
        checkpoint_saver.save_checkpoint(checkpoint, 'iter_' + str(epoch) + '_model.pth.tar', write_best=False)
    else:
        checkpoint_saver.save_checkpoint(checkpoint, 'epoch_' + str(epoch) + '_model.pth.tar', write_best=False)


def write_summary(stats, summary, epoch, mode):
    """
    write the current epoch result to tensorboard
    """
    summary.writer.add_scalar(mode + '/rr', stats['rr'], epoch)
    summary.writer.add_scalar(mode + '/l1', stats['l1'], epoch)
    summary.writer.add_scalar(mode + '/l1_raw', stats['l1_raw'], epoch)
    summary.writer.add_scalar(mode + '/occ_be', stats['occ_be'], epoch)
    summary.writer.add_scalar(mode + '/epe', stats['epe'], epoch)
    summary.writer.add_scalar(mode + '/iou', stats['iou'], epoch)
    summary.writer.add_scalar(mode + '/3px_error', stats['px_error_rate'], epoch)
    try:
        summary.writer.add_scalar(mode + '/disp_change_l1', stats['disp_change_l1'], epoch)
        summary.writer.add_scalar(mode + '/disp_change_epe', stats['disp_change_epe'], epoch)
        summary.writer.add_scalar(mode + '/disp_change_3px_error', stats['disp_change_px_error_rate'], epoch)
    except KeyError:
        pass

