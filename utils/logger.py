# MIT License Copyright (c) 2022 joh-fischer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import yaml
import csv
import os
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Aggregator:
    def __init__(self):
        """ Aggregate sum and average. """
        self.sum = 0.
        self.avg = 0.
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Dummy:
    """ Mock class for tensorboard `SummaryWriter` """
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, item):
        def func(*args, **kwargs):
            pass
        return func


class Logger:
    def __init__(self, log_dir: str = 'logs', exp_name: str = None, timestamp: bool = False,
                 tensorboard: bool = False):
        """
        Custom logger for PyTorch training loops.
        Notes
        ----------
        The logs are stored in <log_dir>/<exp_name>/<timestamp> if `exp_name` is set and `timestamp` is true. For
        example: './logs/model1/22-07-07_121028'. This folder then contains 'metrics.csv' and 'hparams.yaml'.
        Parameters
        ----------
        log_dir : str
            Base directory of experiment logs. Default: 'logs'.
        exp_name : str
            Experiment name to create specific folder. Logs are stored in <log_dir>/<exp_name>.
        timestamp : bool
            If true, makes a folder with the current time for the logs, e.g. <log_dir>/<exp_name>/<timestamp>. The
            timestamp format is `y-m-d_HMS`, e.g. 22-07-07_121028.
                tensorboard : bool
            If true, creates a `SummaryWriter` instance to write to tensorboard.
        """
        self.exp_name = exp_name if exp_name else ''
        self.timestamp = datetime.now().strftime('%y-%m-%d_%H%M%S') if timestamp else ''
        self.log_dir = os.path.join(log_dir, self.exp_name, self.timestamp)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir) if tensorboard else Dummy()

        self.metrics = []
        self.hparams = {}

        self.running_epoch = -1

        self.epoch = {}

        self.global_train_step = 0

    @property
    def tensorboard(self):
        """ Returns tensorboard `SummaryWriter` instance. """
        return self.writer

    def log_hparams(self, params: dict):
        """
        Log hyper-parameters.
        Parameters
        ----------
        params : dict
            Dictionary containing the hyperparameters as key-value pairs. For
            example {'optimizer': 'Adam', 'lr': 1e-02}.
        """
        self.hparams.update(params)

    def log_metrics(self, metrics_dict: dict, step: int = None, phase: str = '', aggregate: bool = False, n: int = 1):
        """
        Log metrics.
        Parameters
        ----------
        metrics_dict : dict
            Dictionary containing the metrics as key-value pairs. For example {'acc': 0.9, 'loss': 0.2}.
        step : int, optional
            Step number where metrics are to be recorded.
        phase : str, optional
            Current phase, e.g. 'train' or 'val' or 'epoch_end'.
        aggregate : bool
            If true, aggregates values into sum and average.
        n : int
            Count for aggregating values.
        """
        step = step if step is not None else len(self.metrics)

        metrics = {'epoch': self.running_epoch, 'phase': phase}
        metrics.update({k: self._handle_value(v) for k, v in metrics_dict.items()})
        metrics['step'] = step

        self.metrics.append(metrics)

        if aggregate:
            for metric_name, metric_val in metrics_dict.items():
                if metric_name not in self.epoch:
                    self.epoch[metric_name] = Aggregator()
                self.epoch[metric_name].update(metric_val, n)

    def init_epoch(self, epoch: int = None):
        """
        Initializes a new epoch.
        Parameters
        ----------
        epoch : int
            If empty, running epoch will be increased by 1.
        """
        if epoch:
            self.running_epoch = epoch
        else:
            self.running_epoch += 1

        self.epoch = {}

    def save(self, name: str = None):
        """
        Save the hyperparameters and metrics to a file.
        Parameters
        ----------
        name : str
            Additional name for saving the metrics and hyperparameters.
        """
        name = '_' + name if name else ''

        if self.metrics:
            metrics_file_path = os.path.join(self.log_dir, 'metrics' + name + '.csv')
            last_m = {}
            for m in self.metrics:
                last_m.update(m)
            metrics_keys = list(last_m.keys())

            with open(metrics_file_path, 'w') as file:
                writer = csv.DictWriter(file, fieldnames=metrics_keys)
                writer.writeheader()
                writer.writerows(self.metrics)

        if self.hparams:
            hparams_file_path = os.path.join(self.log_dir, 'hparams' + name + '.yaml')

            output = yaml.dump(self.hparams, Dumper=yaml.Dumper)
            with open(hparams_file_path, 'w') as file:
                file.write(output)

    @staticmethod
    def _handle_value(value):
        if isinstance(value, torch.Tensor):
            return value.item()
        return value
