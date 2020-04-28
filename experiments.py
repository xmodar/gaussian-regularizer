import json
import math
from argparse import ArgumentParser
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models

__all__ = [
    'tensorboard_logger',
    'flatten_dict',
    'Experiment',
    'CIFAR10Exp',
    'GaussianExp',
]


@contextmanager
def tensorboard_logger(path, override_add_hparams=True):
    try:
        logger = None
        if path:
            # pylint: disable=import-outside-toplevel
            from torch.utils.tensorboard import SummaryWriter
            from torch.utils.tensorboard.summary import hparams
            # pylint: enable=import-outside-toplevel
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            # get the last experiment folder in path
            while True:
                run = 1 + max((int(x.name)
                               for x in path.iterdir()
                               if x.is_dir() and x.name.isdigit()),
                              default=0)
                try:
                    (path / str(run)).mkdir(exist_ok=False)
                    break  # hopefully, you are the first to create the folder
                except FileExistsError:
                    continue
            logger = SummaryWriter(path / str(run))
            if override_add_hparams:

                def add_hparams(hparam_dict=None, metric_dict=None):
                    for summary in hparams(hparam_dict, metric_dict):
                        logger.file_writer.add_summary(summary)

                logger.add_hparams = add_hparams
        yield logger
    finally:
        if path and logger is not None:
            logger.close()


def flatten_dict(any_dict):
    flat = {}
    for k, v in any_dict.items():
        if isinstance(v, dict):
            for fk, fv in flatten_dict(v).items():
                flat[f'{k}/{fk}'] = fv
        else:
            flat[k] = v
    return flat


class Experiment:

    def __init__(self, **config):
        # the initialization order below is important
        self.epoch = 0
        self.config = config
        self.device = self.init_device()
        self.data = self.init_data()
        self.model = self.init_model()
        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_scheduler()

    @staticmethod
    def arg_parser(*args, **kwargs):
        parser = ArgumentParser(*args, **kwargs)

        def add(*a, t=str, d=None, h=None, r=None, **k):
            parser.add_argument(*a, type=t, default=d, help=h, required=r, **k)

        add('--log', h='log directory')
        add('--lr', t=float, r=True, h='learning rate')
        add('--gpu', t=int, h='CUDA device index')
        add('--model', r=True, h='torchvision.models.*')
        add('--dataset', r=True, h='torchvision.datasets.*')
        add('--optimizer', d='SGD', h='torch.optim.*')
        add('--scheduler', h='torch.optim.lr_scheduler.*')
        add('--epochs', t=int, r=True, h='maximum number of epochs')
        add('--patience', t=int, d=float('inf'), h='early stopping')
        add('--weight-decay', t=float, d=0, h='weight decay')
        add('--momentum', t=float, d=0.9, h='momentum')
        return parser

    def init_device(self):
        if torch.cuda.is_available():
            if self.config['gpu']:
                device = torch.device('cuda', self.config['gpu'])
            else:
                empty_model = nn.Sequential()
                device = nn.DataParallel(empty_model).src_device_obj
            self.config['gpu'] = device.index
        else:
            device = torch.device('cpu')
        return device

    def init_data(self):
        # {'train': train_loader, 'test': test_loader}
        raise NotImplementedError(self.config['dataset'])

    def init_model(self):
        # nn.Module().to(self.device)
        raise NotImplementedError(self.config['model'])

    def init_optimizer(self):
        name = self.config['optimizer']
        Optimizer = getattr(torch.optim, name)
        config = dict(
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay'],
        )
        if 'Adam' in name:
            config['betas'] = (self.config['momentum'], 0.999)
        elif name in ('SGD', 'RMSprop'):
            config['momentum'] = self.config['momentum']
        return Optimizer(self.model.parameters(), **config)

    def init_scheduler(self):
        if self.config['scheduler']:
            Scheduler = getattr(lr_scheduler, self.config['scheduler'])
            return Scheduler(self.optimizer)
        return None

    @property
    def learning_rate(self):
        return max(p['lr'] for p in self.optimizer.param_groups)

    def state_dict(self):
        state_dict = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if self.scheduler:
            state_dict['scheduler'] = self.scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if state_dict.get('scheduler'):
            self.scheduler.load_state_dict(state_dict['scheduler'])

    def step_batch(self, batch):
        if isinstance(batch, list):
            batch = {'inputs': batch[0], 'targets': batch[1]}
        assert 'size' not in batch, '`size` is a key reserved for batch_size'
        batch['size'] = batch['inputs'].size(0)
        batch['inputs'] = batch['inputs'].to(self.device)
        batch['targets'] = batch['targets'].to(self.device)
        return batch

    def step_outputs(self, batch, train):
        logits = self.model.forward(batch['inputs'])
        loss = F.cross_entropy(logits, batch['targets'])
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return {'logits': logits.detach(), 'loss': loss.item()}

    def step_scheduler(self, metrics):
        if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(self.train_stopping_value(metrics), self.epoch)
        elif self.scheduler is not None:
            self.scheduler.step(self.epoch)

    def step_metrics(self, metrics=None, batch=None, outputs=None):
        if not metrics:  # initialize
            return {'count': 0, 'loss': 0, 'accuracy': 0}
        if None not in (batch, outputs):  # update
            metrics['count'] += batch['size']
            metrics['loss'] += float(outputs['loss']) * batch['size']
            pred = outputs['logits'].data.argmax(1)
            metrics['accuracy'] += (pred == batch['targets']).sum().item()
        else:  # final averaging
            count = metrics.pop('count')
            metrics['loss'] /= count
            metrics['accuracy'] /= count

    def step(self, train=True):
        self.model.train(train)
        with torch.set_grad_enabled(train):
            metrics = self.step_metrics()  # initialize
            loader = self.data['train'] if train else self.data['test']
            for batch in loader:
                batch = self.step_batch(batch)
                outputs = self.step_outputs(batch, train)
                self.step_metrics(metrics, batch, outputs)  # update
            self.step_metrics(metrics)  # final averaging
            if train:
                metrics['epoch'] = self.epoch
                metrics['lr'] = self.learning_rate
                metrics = {'train': metrics, 'test': self.step(False)}
                self.step_scheduler(metrics)
                self.epoch += 1
        return metrics

    def train_logger(self, path):
        return tensorboard_logger(path)

    def train_save(self, path, metrics=None):
        path = Path(path)
        config_path = path / 'config.json'
        if not config_path.exists():
            with open(config_path, 'w') as config_file:
                json.dump(self.config, config_file)
        if metrics:
            with open(path / 'metrics.json', 'w') as metrics_file:
                json.dump(metrics, metrics_file)
        torch.save(self.state_dict(), path / 'state_dict.pt')

    def train_log_progress(self, logger, metrics, verbose=True):
        if verbose:
            epoch = metrics['train']['epoch']
            lr = metrics['train']['lr']
            trn_l = metrics['train']['loss']
            trn_a = metrics['train']['accuracy']
            tst_l = metrics['test']['loss']
            tst_a = metrics['test']['accuracy']
            print(f'{epoch:05d} @ {lr:10.4e} '
                  f'Train[{trn_l:11.4e}x{trn_a * 100:6.2f}%] '
                  f'Test[{tst_l:11.4e}x{tst_a * 100:6.2f}%]')
        if logger:
            for k, v in flatten_dict(metrics).items():
                logger.add_scalar(k, v, self.epoch)

    def train_log_hparams(self, logger, metrics, verbose=True):
        if verbose:
            print('Best epoch:')
            self.train_log_progress(None, metrics, verbose)
        if logger:
            logger.add_hparams(self.config, flatten_dict(metrics))

    def train_early_stopping(self, best, metrics):
        patience = self.config.get('patience', float('inf'))
        if self.epoch - best['epoch'] > patience:
            print('Early stopping, exceeded patience!')
            return True
        if math.isnan(metrics['train']['loss']):
            print('Early stopping, reached NaN loss!')
            return True
        return False

    def train_stopping_value(self, metrics):
        return metrics['test']['loss']

    def train_stopping(self, best=None, logger=None, metrics=None):
        if not best:  # initialize
            return {'epoch': 0, 'value': float('inf')}
        if metrics:  # log and check
            value = self.train_stopping_value(metrics)
            if value <= best['value']:
                best['epoch'] = self.epoch
                best['value'] = value
                best['metrics'] = metrics
                if logger:
                    self.train_save(logger.log_dir, metrics)
            return self.train_early_stopping(best, metrics)
        else:  # done training
            self.train_log_hparams(logger, best.get('metrics', {}))

    def train(self):
        if self.epoch >= self.config['epochs']:
            return
        with self.train_logger(self.config['log']) as logger:
            if logger is not None:
                self.config['path'] = str(Path(logger.log_dir).absolute())
            best = self.train_stopping()  # initialize
            for _ in range(self.epoch, self.config['epochs']):
                metrics = self.step()
                self.train_log_progress(logger, metrics)
                if self.train_stopping(best, logger, metrics):  # log and check
                    break
            self.train_stopping(best, logger)  # done training


class CIFAR10Exp(Experiment):

    @staticmethod
    def arg_parser(*args, **kwargs):
        parser = Experiment.arg_parser(*args, **kwargs)
        parser.add_argument('--train-batch', type=int, default=500)
        parser.add_argument('--test-batch', type=int, default=1000)
        defaults = {
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'momentum': 0.9,
            'epochs': 50,
            'model': 'vgg16_bn',
            'dataset': 'CIFAR10',
            'optimizer': 'Adam',
            'scheduler': 'ReduceLROnPlateau',
        }
        parser.set_defaults(**defaults)
        for action in parser._actions:  # pylint: disable=protected-access
            if action.dest in defaults:
                action.required = False
        return parser

    def init_data(self):
        name = self.config['dataset']
        Data = getattr(datasets, name)
        if name != 'FakeData':
            root = Path.home() / f'.torch/datasets/{name}'
            root.mkdir(parents=True, exist_ok=True)
        if name == 'CIFAR10':
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4915, 0.4823, 0.4468),
                            (0.2470, 0.2435, 0.2616)),
            ])
            train_transform = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(p=0.5),
                *transform.transforms,
            ])
            train_set = Data(root, True, train_transform, download=True)
            test_set = Data(root, False, transform)
            jobs = dict(pin_memory=self.device.type == 'cuda', num_workers=4)
            r_batch = self.config['train_batch']
            t_batch = self.config['test_batch']
            data = {
                'train': DataLoader(train_set, r_batch, shuffle=True, **jobs),
                'test': DataLoader(test_set, t_batch, **jobs),
            }
        else:
            raise NotImplementedError(self.config['dataset'])
        return data

    def init_model(self):
        name = self.config['model']
        if name.startswith('vgg'):
            Model = getattr(models, name)
            model = Model()
            model = nn.Sequential(
                *model.features,
                # model.avgpool,
                nn.Flatten(1),
                # *model.classifier,
                nn.Linear(512, 10),
            )
            # switch ReLU>MaxPool2d to MaxPool2d>ReLU for efficiency
            for i, (a, b) in enumerate(zip(model, model[1:])):
                if isinstance(a, nn.ReLU) and isinstance(b, nn.MaxPool2d):
                    model[i], model[i + 1] = model[i + 1], model[i]
        elif name.startswith('resnet'):
            from cifar_resnet import ResNet
            depth = ''.join(c for c in name if c.isdigit())
            model = ResNet(depth=int(depth), num_classes=10)
            model = nn.Sequential(
                model.conv1,
                model.bn1,
                nn.ReLU(inplace=True),
                model.layer1,
                model.layer2,
                model.layer3,
                model.avgpool,
                nn.Flatten(1),
                model.fc,
            )
        elif name == 'alexnet':
            model = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Flatten(1),
                nn.Linear(256, 10),
            )
        else:
            raise NotImplementedError(self.config['model'])
        return model.to(self.device)

    def train_stopping_value(self, metrics):
        return -metrics['test']['accuracy']


class GaussianExp(CIFAR10Exp):

    def __init__(self, **config):
        assert config['sigma'] >= 0
        assert 0 <= config['gamma'] <= 1
        assert 0 <= config['alpha'] <= 1
        if config['sigma'] > 0:
            assert config['gamma'] > 0 or config['alpha'] > 0
        else:
            assert config['gamma'] == config['alpha'] == 0
        super().__init__(**config)

    @staticmethod
    def arg_parser(*args, **kwargs):
        p = CIFAR10Exp.arg_parser(*args, **kwargs)
        loss = ['cross_entropy', 'cross_entropy_outputs', 'consistency']
        p.add_argument('--moment-loss', choices=loss, default='cross_entropy')
        p.add_argument('--sigma', type=float, default=0, help='noise level')
        p.add_argument('--gamma', type=float, default=0, help='augmentation')
        p.add_argument('--alpha', type=float, default=0, help='trade-off')
        return p

    def init_model(self):
        model = super().init_model()
        assert isinstance(model, nn.Sequential)
        assert isinstance(model[0], nn.Conv2d)
        if isinstance(model[1], nn.BatchNorm2d):
            assert isinstance(model[2], nn.ReLU)
        else:
            assert isinstance(model[1], nn.ReLU)
        return model

    @property
    def sigma(self):
        std = self.data['train'].dataset.transform.transforms[-1].std
        std = torch.tensor(std, dtype=torch.float32, device=self.device)
        return float(self.config['sigma']) / std.view(-1, 1, 1)

    def step_batch(self, batch):
        batch = super().step_batch(batch)
        if self.config['gamma'] > 0:
            noise = torch.randn_like(batch['inputs']) * self.sigma
            p = 1 - self.config['gamma']
            if p > 0:
                F.dropout2d(noise.unsqueeze(0), p, inplace=True).squeeze(0)
            batch['inputs'] += noise
        return batch

    @staticmethod
    def conv2d_var_mean(conv, var, mu):
        # var = F.conv2d(torch.ones_like(mu) * var, conv.weight**2, pad...)
        # for efficiency, we will approximate this by assuming no padding
        # only border activations will be erroneous which is not that bad
        w = conv.weight
        var = (w * w * var).flatten(1).sum(1).view(-1, 1, 1)
        return var, conv(mu)

    @staticmethod
    def batch_norm_var_mean(bn, var, mu):
        if bn.track_running_stats:
            r_var = bn.running_var
        else:
            r_var = mu.transpose(1, 0).flatten(1).var(1, unbiased=False)
        var = var / (r_var.view(-1, 1, 1) + bn.eps)
        if bn.affine:
            var = var * (bn.weight * bn.weight).view(-1, 1, 1)
        return var, bn(mu)

    @staticmethod
    def relu_mean(std, mu):
        mu_std = mu / (std * math.sqrt(2))
        cdf = 0.5 + 0.5 * mu_std.erf()
        pdf = math.sqrt(0.5 / math.pi) * (-mu_std * mu_std).exp()
        return mu * cdf + std * pdf

    def gaussian_forward(self, batch, train):
        var, mu = self.sigma * self.sigma, batch['inputs']
        var, mu = self.conv2d_var_mean(self.model[0], var, mu)

        i = 1
        if isinstance(self.model[i], nn.BatchNorm2d):
            var, mu = self.batch_norm_var_mean(self.model[i], var, mu)
            i += 1

        rest = self.model[i + 1:]
        mu_forward = rest(self.relu_mean(var.sqrt(), mu))

        if train and self.config['alpha'] == 1:
            with torch.no_grad():
                rest.eval()
                forward = rest(mu.detach().clamp(min=0))
                rest.train()
        else:
            forward = rest(mu.clamp(min=0))

        return mu_forward, forward

    def step_outputs(self, batch, train):
        a = self.config['alpha']
        if a == 0:
            return super().step_outputs(batch, train)
        mu_logits, logits = self.gaussian_forward(batch, train)
        empirical_loss = F.cross_entropy(logits, batch['targets'])
        loss_type = self.config['moment_loss']
        if loss_type == 'cross_entropy':
            moment_loss = F.cross_entropy(mu_logits, batch['targets'])
        elif loss_type == 'cross_entropy_outputs':
            moment_loss = F.cross_entropy(mu_logits, logits.argmax(1))
        elif loss_type == 'consistency':
            moment_loss = F.mse_loss(mu_logits, logits.detach())
        else:
            raise ValueError(f'Unknown moment loss type: {loss_type}')
        loss = (1 - a) * empirical_loss + a * moment_loss

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        outputs = {
            'logits': logits.detach(),
            'moment_logits': mu_logits.detach(),
            'loss': loss.item(),
            'moment_loss': moment_loss.item(),
            'empirical_loss': empirical_loss.item(),
        }
        return outputs

    def step_metrics(self, metrics=None, batch=None, outputs=None):
        if self.config['alpha'] == 0:
            return super().step_metrics(metrics, batch, outputs)
        if not metrics:  # initialize
            metrics = super().step_metrics()
            metrics['moment_loss'] = 0
            metrics['empirical_loss'] = 0
            metrics['moment_accuracy'] = 0
            return metrics
        if None not in (batch, outputs):  # update
            super().step_metrics(metrics, batch, outputs)
            batch_size = batch['size']
            metrics['moment_loss'] += outputs['moment_loss'] * batch_size
            metrics['empirical_loss'] += outputs['empirical_loss'] * batch_size
            p = outputs['moment_logits'].data.argmax(1)
            metrics['moment_accuracy'] += (p == batch['targets']).sum().item()
        else:  # final averaging
            count = metrics['count']
            metrics['moment_loss'] /= count
            metrics['empirical_loss'] /= count
            metrics['moment_accuracy'] /= count
            super().step_metrics(metrics)


if __name__ == '__main__':
    GaussianExp(**vars(GaussianExp.arg_parser().parse_args())).train()
