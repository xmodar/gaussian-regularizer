import os
import sys
import torch
from glob import glob
import tensorflow as tf
from shutil import rmtree
from functools import wraps
from itertools import product
from traceback import print_exc
import network_moments.torch.gaussian as gnm
from argparse import Namespace, ArgumentParser
from collections import namedtuple, OrderedDict

FILE = 'args.txt'

__all__ = ['GNM', 'Trainer', 'ARGS', 'main']


class GNM:
    '''GNM expects the first layer to be a conv2d followed by a relu.'''

    def moments(self, mu, var, output=True, mean=True):
        '''Compute the forward pass with the moments of the network.

        Computes the forward pass and the output mean.
        Approximates the output mean of the network for Gaussian input
        by linearizing around the first convolutional layer.

        Args:
            mu: Input mean (Batch, 1, 28, 28).
            var: The input variance (1, 28, 28) or a scalar.

        Returns:
            (forward: The forward pass,
             mean: The output mean of the network).
        '''
        if not mean:
            return self.forward(mu), None
        layer = self[0]
        if not torch.is_tensor(var):
            var = torch.tensor(var, dtype=mu.dtype, device=mu.device)
        w = layer.weight
        affine_mu = layer(mu)
        if var.numel() == 1:
            var = var.repeat(1, *mu.shape[1:])
        else:
            var = var.view(1, *mu.shape[1:])
        affine_var = torch.nn.functional.conv2d(var, w**2,
                                                stride=layer.stride,
                                                padding=layer.padding,
                                                dilation=layer.dilation,
                                                groups=layer.groups)
        relu_mu = gnm.relu.mean(affine_mu, affine_var)
        out_mean = self.forward(relu_mu, layers=self[2:])
        if not output:
            return None, out_mean
        forward = self.forward(affine_mu.clamp(min=0), layers=self[2:])
        return forward, out_mean


class LeNetGNM(GNM, gnm.net.LeNet):
    @wraps(gnm.net.LeNet.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # swiching relu and maxpool doesn't affect the output of the network
        self[1], self[2] = self[2], self[1]


class AlexNetGNM(GNM, gnm.net.AlexNet):
    @wraps(gnm.net.AlexNet.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self[1], self[2] = self[2], self[1]
        del self[17], self[14]  # remove dropouts


class VGG16GNM(GNM, gnm.net.VGG16):
    @wraps(gnm.net.VGG16.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # remove dropouts and the first batch_norm
        if isinstance(self[1], torch.nn.BatchNorm2d):
            del self[50], self[47], self[1]
        else:
            del self[37], self[34]


class Trainer(gnm.net.ClassifierTrainer):
    network = {
        'lenet': LeNetGNM,
        'alexnet': AlexNetGNM,
        'vgg16': VGG16GNM,
    }

    @classmethod
    def loss(cls, model, data, target, optimizer):
        config = model.config.optimization
        if not hasattr(config, 'scaled_input_variance'):
            mu = model.input_mean
            var = config.input_variance
            input_range = float(2 * (len(mu) - sum(mu)) / sum(model.input_std))
            config.scaled_input_variance = var / input_range ** 2
        coef = config.loss_terms
        mean = coef.expectation != 0 or not cls.ignore_zero_loss_coefficients
        phase = torch.no_grad if optimizer is None else torch.enable_grad
        with phase():
            output, expectation = model.moments(
                data, config.scaled_input_variance, mean=mean)
        terms = {
            'empirical': Namespace(
                coef=coef.empirical, func=cls.softmax_cross_entropy,
                args=(output, target), kwargs={},
            ),
            'expectation': Namespace(
                coef=coef.expectation, func=cls.softmax_cross_entropy,
                args=(expectation, target), kwargs={},
            ),
        }
        metrics = {
            'accuracy': cls.count_correct(output.data, target).item()
        }
        return terms, metrics

    @classmethod
    def test(cls, config):
        # get the model
        model = config.model.network(**vars(config.model.config))
        model = model.to(config.device)
        model.config = config
        state = torch.load(config.checkpoint)
        model.load_state_dict(state['model'])

        # compute the test accuracy
        performance = getattr(model, config.model.metric_function)
        loader = cls.data_from_config(config, model=model, train=False)
        test_accuracy = performance(loader, config.device)

        # compute the robustness of the model
        robustness, plot = model.gaussian_robustness(loader, config.device)

        return state['epoch'], test_accuracy, robustness, plot

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.optimization.config.weight_decay = 0
        config.optimization.loss_terms.expectation = 0
        config.optimization.input_variance = 0.125 ** 2
        return config

    @classmethod
    def experiment(cls, model, dataset, empirical,
                   sigma, augmentation, expectation,
                   epochs=100, device='cuda', name='poc', run=False):
        # get the configurations of this experiment
        config = cls.config(model, dataset, empirical,
                            sigma, augmentation, expectation,
                            epochs, device, name)
        if config is None:
            return
        done = cls.finished(config)
        status = 'done' if done else 'todo'
        folder = os.path.basename(config.log_dir)
        state = f'[{status}] {model} - {dataset} [{name}]: {folder}'
        if not run:
            return state
        if done:
            return
        cls.delete_experiment(config)

        # redirect stdout and stderr to log files
        log_file = os.path.join(config.log_dir, 'log.')
        with cls.Tee(log_file + 'out', log_file + 'err'):
            # train the model according to config
            print(state)
            cls.train_from_config(config)

            # test the model's accuracy and robustness
            epoch, test_accuracy, robustness, plot = cls.test(config)
            print('Test accuracy = {:.2f}%'.format(100 * test_accuracy))
            print(f'Achieved robustness = {100 * robustness:.2f}%')

            # log the rebustness results into tensorboard
            if config.log_dir is not None:
                def value(k, v):
                    return tf.Summary.Value(tag=k, simple_value=v)
                directory = os.path.join(config.log_dir, f'epoch_{epoch}')
                writer = tf.summary.FileWriter(directory)
                writer.add_summary(tf.Summary(value=[
                    value('test/accuracy', test_accuracy),
                    value('test/robustness', robustness),
                ]), epoch)
                for i, (sig, acc) in enumerate(zip(*plot)):
                    writer.add_summary(tf.Summary(value=[
                        value('test/robustness/sigmas', sig),
                        value('test/robustness/accuracies', acc),
                    ]), i)
                writer.flush()
                writer.close()

    @classmethod
    def finished(cls, config):
        log_file = os.path.join(config.log_dir, 'log.out')
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                content = [line.strip() for line in f.readlines()]
                if len(content) > 0 and 'Achieved' in content[-1]:
                    return True
        return False

    @classmethod
    def delete_experiment(cls, config):
        if config.log_dir is not None:
            if os.path.exists(config.log_dir):
                rmtree(config.log_dir)
        if config.checkpoint is not None:
            if os.path.exists(config.checkpoint):
                os.remove(config.checkpoint)

    @classmethod
    def config(cls, model, dataset, empirical,
               sigma, augmentation, expectation,
               epochs=100, device='cuda', name='default'):
        if (empirical == 0 and expectation != 1):
            return
        if expectation == 0 and empirical != 1:
            return
        if (sigma == 0) != (expectation == 0 and augmentation == 0):
            return
        if augmentation != 0 and (expectation != 0 or sigma > 1):
            return
        config = cls.default_config()
        config.epochs = epochs
        config.device = device
        config.optimization.input_variance = sigma ** 2
        config.optimization.loss_terms.empirical = empirical
        config.optimization.loss_terms.expectation = expectation
        name = os.path.join(name, f'emp_{empirical:.4e}_sig_{sigma:.4e}')
        if augmentation == 0:
            name += f'_exp_{expectation:.4e}'
        else:
            name += f'_aug_{augmentation:03d}'
        cls.config_model_dataset(config, model, dataset, name)
        if config.model.config.input_size < config.model.network.min_size:
            return
        config.data.train.augmentation = augmentation

        # model-dataset specific options
        if model != 'lenet' and dataset == 'mnist':
            return
        if model != 'lenet':
            config.patience = 20
            if model == 'alexnet':
                config.data.train.batch_size = 128
            if model == 'vgg16':
                config.data.train.batch_size = 256
                config.data.valid.batch_size = 1000
            config.optimization.optimizer = torch.optim.SGD
            config.optimization.config = Namespace(
                lr=0.1,
                dampening=0,
                momentum=0.9,
                weight_decay=5e-4,
                nesterov=True,
            )
            sch = config.lr_scheduling.config
            sch.factor = 0.5
            sch.patience = 5

        return config

    @classmethod
    def data_from_config(cls, config, train=False, model=None):
        out = super().data_from_config(config, train=train, model=model)
        folds = config.data.train.augmentation + 1
        if not train or folds == 1:
            return out
        train_loader, valid_loader = out
        original_len = train_loader.dataset.__len__()
        len_dataset = original_len * folds
        sigma = config.optimization.input_variance ** 0.5
        seed = int(torch.randint(sys.maxsize - folds, tuple()))
        original_getitem = train_loader.dataset.__getitem__

        class Augmented(torch.utils.data.Dataset):
            def __getitem__(self, index):
                if not -len_dataset <= index < len_dataset:
                    raise IndexError(f'{index} not in size {len_dataset}')
                index = index % len_dataset
                image, label = original_getitem(index % original_len)
                offset = index // original_len
                if offset == 0:
                    return image, label
                with gnm.utils.rand.RNG(seed + offset, devices=[image.device]):
                    new_image = torch.randn_like(image).mul_(sigma).add_(image)
                return new_image, label

            def __len__(self):
                return len_dataset

        new_train_set = Augmented()
        train_loader.dataset = new_train_set
        train_loader.sampler.data_source = new_train_set

        return train_loader, valid_loader

    class Tee:
        def __init__(self, stdout, stderr, append=False):
            '''Mirror the output and error streams to files.'''
            def filestream(path, stream):
                directory = os.path.dirname(os.path.abspath(path))
                if not os.path.exists(directory):
                    os.makedirs(directory)
                file = open(path, 'a' if append else 'w')
                file.original_stream = stream
                _write = file.write

                def write(message):
                    stream.write(message)
                    _write(message)
                file.write = write
                _flush = file.flush

                def flush():
                    stream.flush()
                    _flush()
                file.flush = flush
                return file

            self.stdout = filestream(stdout, sys.stdout)
            self.stderr = filestream(stderr, sys.stderr)

        def __enter__(self):
            sys.stdout = self.stdout
            sys.stderr = self.stderr

        def __exit__(self, exception, instance, traceback):
            try:
                self.stdout.flush()
                self.stderr.flush()
            finally:
                try:
                    self.stdout.close()
                    self.stderr.close()
                finally:
                    sys.stdout = self.stdout.original_stream
                    sys.stderr = self.stderr.original_stream

    class Results:
        scalar_summary = namedtuple('scalar_summary',
                                    ('value', 'step', 'time'))

        @classmethod
        def all_experiments(cls, root='exps'):
            return glob(os.path.join(root, '*/*/*/*'))

        @classmethod
        def delete_incomplete_experiments(cls, root='exps', net_root='models'):
            deleted = []
            nets = glob(os.path.join(net_root, '*/*/*'))
            keys = ['[*]'.join(cls.split(n[:n.rfind('.')])[1:]) for n in nets]
            for exp in cls.all_experiments(root):
                try:
                    cls.raw_results(exp)
                except:
                    rmtree(exp)
                    key = cls.split(exp)[1:]
                    key[1] = key[1] + '_' + key.pop(2)
                    key = '[*]'.join(key)
                    if key in keys:
                        os.remove(nets[keys.index(key)])
                    deleted.append(exp)
            return deleted

        @classmethod
        def events_files(cls, root):
            out = glob(os.path.join(root, 'events*'))
            epochs = [int(d[d.rfind('_') + 1:])
                      for d in os.listdir(root) if d.startswith('epoch_')]
            out += glob(os.path.join(root, f'epoch_{max(epochs)}', 'events*'))
            return out

        @classmethod
        def config(cls, experiment):
            res = cls.arguments(experiment)
            return Trainer.config(res['model'], res['dataset'], res['emp'],
                                  res['sig'], res['aug'], res['exp'],
                                  name=res['name'])

        @classmethod
        def split(cls, path):
            chunks = []
            while path:
                path, name = os.path.split(path)
                chunks.insert(0, name)
            return chunks

        @classmethod
        def arguments(cls, experiment):
            path, name = os.path.split(experiment)
            exp = name.split('_')
            out = {exp[2 * i]: float(exp[2 * i + 1])
                   for i in range(len(exp) // 2)}
            if 'exp' not in out:
                out['exp'] = 0
            if 'aug' not in out:
                out['aug'] = 0
            out['aug'] = int(out['aug'])
            out['dataset'], out['model'], out['name'] = cls.split(path)[-3:]
            return out

        @classmethod
        def parse_events_files(cls, experiment):
            summaries = {}
            for events_file in cls.events_files(experiment):
                for event in tf.train.summary_iterator(events_file):
                    for value in event.summary.value:
                        key = value.tag
                        if key not in summaries:
                            summaries[key] = []
                        summary = cls.scalar_summary(
                            value.simple_value, event.step, event.wall_time)
                        summaries[key].append(summary)
            return summaries

        @classmethod
        def all_results(cls, root='exps', raw=False):
            for experiment in cls.all_experiments(root):
                try:
                    if raw:
                        yield cls.raw_results(experiment)
                    else:
                        yield cls.results(experiment)
                except:
                    pass

        @classmethod
        def raw_results(cls, experiment):
            out = cls.arguments(experiment)
            out['summary'] = cls.parse_events_files(experiment)
            return out

        @classmethod
        def clean_summary(cls, summary):
            epoch = summary['test/accuracy'][0].step
            times = [v.time for v in summary['train/loss'] if v.step <= epoch]
            relative = [t1 - t2 for t1, t2 in zip(times[1:], times[:-1])]

            def values(key, return_all=False):
                if key not in summary:
                    return []
                return [v.value for v in summary[key]
                        if return_all or v.step <= epoch]

            def phase_summary(phase):
                return Namespace(
                    accuracy=values(f'{phase}/accuracy'),
                    loss=values(f'{phase}/loss'),
                    loss_terms=Namespace(
                        empirical=values(f'{phase}/loss/empirical'),
                        expectation=values(f'{phase}/loss/expectation'),
                    ),
                )

            out = Namespace(
                last_epoch=epoch,
                learning_rate=values('learning_rate'),
                time=relative,
                train=phase_summary('train'),
                valid=phase_summary('valid'),
                test=Namespace(
                    accuracy=summary['test/accuracy'][0].value,
                    robustness=summary['test/robustness'][0].value,
                    robustness_plot=Namespace(
                        sigmas=values('test/robustness/sigmas', True),
                        accuracies=values('test/robustness/accuracies', True),
                    ),
                ),
            )
            return out

        @classmethod
        def results(cls, experiment):
            out = cls.raw_results(experiment)
            out['summary'] = cls.clean_summary(out['summary'])

            # correct the old robustness evaluation
            test = out['summary'].test
            if len(test.robustness_plot.sigmas) == 0:
                corrected = os.path.join(experiment, 'robustness.pt')
                if not os.path.exists(corrected):
                    config = cls.config(experiment)
                    if config is None:
                        print('Could\'t retrieve config for: ', experiment)
                        return out
                    model = Trainer.model_from_config(config, True)
                    loader = Trainer.data_from_config(config, False, model)
                    rob, plt = model.gaussian_robustness(loader, config.device)
                    torch.save({
                        'robustness': float(rob),
                        'sigmas': plt[0].cpu().numpy().tolist(),
                        'accuracies': plt[1].cpu().numpy().tolist(),
                    }, corrected)
                corrected = torch.load(corrected)
                test.robustness = corrected['robustness']
                test.robustness_plot.sigmas = corrected['sigmas']
                test.robustness_plot.accuracies = corrected['accuracies']
            return out


ARGS = OrderedDict(
    models=list(Trainer.network.keys()),
    datasets=list(Trainer.dataset.keys()),
    emps=[1],
    sigmas=[0, 0.125, 0.25, 0.325, 0.5, 1, 2, 5, 10, 20],
    augs=[0, 1, 5, 10, 20],
    exps=[0, 0.5, 1, 1.5, 2, 5, 10, 20],
)
# ARGS['models'].pop(ARGS['models'].index('vgg16'))


def get_experiments(device_index, multiple_gpus):
    i = -1
    count = torch.cuda.device_count()
    for args in product(*ARGS.values()):
        state = Trainer.experiment(*args, run=False)
        if state is None:
            continue
        i += 1
        if multiple_gpus and i % count != device_index:
            continue
        yield args, state


def maybe_number(string):
    if string.isdecimal():
        return int(string)
    try:
        return float(string)
    except:
        return string[1:-1]


def run(gpu, args):
    try:
        device = f'cuda:{gpu}'
        Trainer.experiment(*args, device=device, run=True)
    except:
        print_exc()


def main(mode, config):
    if hasattr(config, 'path') and config.path is None:
        config.path = FILE
    if mode != 'run':
        exps = get_experiments(config.gpu, config.multiple)
        todos = filter(lambda x: x[1].startswith('[todo]'), exps)
    if mode == 'show':
        if config.summary:
            done = 0
            for count, (_, state) in enumerate(exps, 1):
                if state.startswith('[done]'):
                    done += 1
            print(f'Done: {done} Todo: {count - done}')
        else:
            for _, state in exps:
                print(state)
    elif mode == 'all':
        for args, state in todos:
            run(config.gpu, args)
    elif mode == 'dump':
        with open(config.path, 'w') as f:
            for args, state in todos:
                f.write(str(args) + '\n')
    elif mode == 'file':
        i = config.experiment
        with open(config.path, 'r') as f:
            lines = f.readlines()
        for line in lines if i is None else [lines[i]]:
            run(config.gpu, [maybe_number(a) for a in line[1:-2].split(', ')])
    elif mode == 'run':
        Trainer.experiment(run=True, **vars(config))
    else:
        raise ValueError(f'Unknown mode: {mode}')


if __name__ == '__main__':
    # define all arguments
    arg = lambda *args, **kwargs: (args, kwargs)
    gpu = arg('-g', '--gpu', default=0, type=int,
              help='The device index of the GPU.')
    multiple = arg('-m', '--multiple', action='store_true',
                   help='Whether in multiple GPUs setup.')
    summary = arg('-s', '--summary', action='store_true',
                  help='Whether to print stats summary or full list.')
    path = arg('-p', '--path', type=str,
               help=(f'Which file to read/dump the '
                     f'arguments list from (default: {FILE}).'))
    experiment = arg('-e', '--experiment', type=int,
                     help='The index to run from file.')

    # add all arguments
    def subparser(name, *prsr_args, description=None):
        prsr = sub.add_parser(name, description=description)
        prsr.name = name
        for a in prsr_args:
            prsr.add_argument(*a[0], **a[1])
        return prsr
    parser = ArgumentParser(description='PoC: Training with expectation')
    sub = parser.add_subparsers(dest='subparser')
    subparser('all', multiple, gpu,
              description='Run the remaining experiments')
    default = subparser('show', summary, multiple, gpu,
                        description='Show the experiments')
    subparser('dump', path, multiple, gpu,
              description='Output to file the remaining experiments')
    subparser('file', path, experiment, multiple, gpu,
              description='Run experiments from file')
    subparser('run',
              arg('-m', '--model', required=True,
                  choices=list(Trainer.network.keys())),
              arg('-d', '--dataset', required=True,
                  choices=list(Trainer.dataset.keys())),
              arg('-emp', '--empirical', type=float, default=1,
                  help='Coefficient of the empirical loss.'),
              arg('-sig', '--sigma', type=float, default=0,
                  help='Input standard deviation.'),
              arg('-aug', '--augmentation', type=int, default=0,
                  help='Number of data augmentation folds.'),
              arg('-exp', '--expectation', type=float, default=0,
                  help='Coefficient of the expectation loss.'),
              arg('-e', '--epochs', type=int, default=100,
                  help='Number of training epochs.'),
              arg('-on', '--device', type=str, default='cuda',
                  help='Which device to run the experiments on.'),
              arg('-n', '--name', type=str, default='default',
                  help='The name of the experiment.'),
              description='Run a specific experiment')

    # parse the arguments
    args = parser.parse_args()
    mode = args.subparser
    delattr(args, 'subparser')
    if mode is None:
        mode = default.name
        args = default.parse_args()

    main(mode, args)
