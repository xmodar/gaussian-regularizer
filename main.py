import os
import sys
import torch
from glob import glob
import tensorflow as tf
from shutil import rmtree
from functools import wraps
from itertools import product
import network_moments.torch.gaussian as gnm
from argparse import Namespace, ArgumentParser


class LeNetGNM(gnm.net.LeNet):
    @wraps(gnm.net.LeNet)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # put relu then maxpool to compute self.mean()
        self[1], self[2] = self[2], self[1]

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


class LeNetGNMTrainer(gnm.net.ClassifierTrainer):
    @classmethod
    def loss(cls, model, data, target, optimizer):
        config = model.config.optimization
        if not hasattr(config, 'scaled_input_variance'):
            mu = model.input_mean
            var = config.input_variance
            input_range = float(2 * (len(mu) - sum(mu)) / sum(model.input_std))
            config.scaled_input_variance = var / input_range ** 2
        coef = config.loss_terms
        mean = not (coef.expectation == 0 and coef.ignore_zeros)
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
    def experiment(cls, dataset, empirical, sigma, expectation,
                   epochs=100, device='cuda'):
        # get the configurations of this experiment
        config = cls.config(dataset, empirical, expectation,
                            sigma, epochs, device)
        if config is None or cls.finished(config):
            return
        cls.delete_experiment(config)

        # redirect stdout and stderr to log files
        log_file = os.path.join(config.log_dir, 'log.')
        with Tee(log_file + 'out', log_file + 'err'):
            # train the model according to config
            name = os.path.basename(config.log_dir)
            print(f'{dataset} on {device} for {epochs} epochs: {name}')
            best_state = cls.train_from_config(config)

            # test the model's accuracy and robustness
            epoch, test_accuracy, robustness, plot = cls.test(config)
            print('Test accuracy = {:.2f}%'.format(100 * test_accuracy))
            print(f'Achieved robustness = {100 * robustness:.2f}%')

            # log the rebustness results into tensorboard
            if config.log_dir is not None:
                value = lambda k, v: tf.Summary.Value(tag=k, simple_value=v)
                directory = os.path.join(config.log_dir, f'epoch_{epoch}')
                writer = tf.summary.FileWriter(directory)
                writer.add_summary(tf.Summary(value=[
                    value('test/accuracy', test_accuracy),
                    value('test/robustness', robustness),
                ]), epoch)
                for i, (var, acc) in enumerate(zip(*plot)):
                    writer.add_summary(tf.Summary(value=[
                        value('test/robustness/variances', var),
                        value('test/robustness/accuracies', acc),
                    ]), i)
                writer.flush()
                writer.close()

        return test_accuracy, robustness, plot

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
    def train_from_config(cls, config):
        ignore = config.optimization.loss_terms.ignore_zeros
        old_ignore = cls.ignore_zero_loss_coefficients
        cls.ignore_zero_loss_coefficients = ignore
        result = super().train_from_config(config)
        cls.ignore_zero_loss_coefficients = old_ignore
        return result

    @classmethod
    def config_model_dataset(cls, config, model='lenet_gnm', dataset='mnist'):
        model_name = 'lenet' if model == 'lenet_gnm' else model
        super().config_model_dataset(config, model=model_name, dataset=dataset)
        if model == 'lenet_gnm':
            config.model.network = LeNetGNM
        config.log_dir = 'poc'.join(config.log_dir.rsplit('default', 1))
        config.checkpoint = 'poc'.join(config.checkpoint.rsplit('default', 1))

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.optimization.config.weight_decay = 0
        config.optimization.input_variance = 0.125 ** 2
        config.optimization.loss_terms = Namespace(
            empirical=1, expectation=0, ignore_zeros=True)
        return config

    @classmethod
    def config(cls, dataset, empirical, sigma, expectation,
               epochs=100, device='cuda'):
        if (empirical == 0 and expectation != 1):
            return
        if expectation == 0 and empirical != 1:
            return
        if (sigma == 0) != (expectation == 0):
            return
        name = f'emp_{empirical:.4e}_sig_{sigma:.4e}_exp_{expectation:.4e}'
        config = cls.default_config()
        cls.config_model_dataset(config, dataset=dataset)
        config.epochs = epochs
        config.device = device
        config.log_dir = os.path.join(config.log_dir, name)
        config.checkpoint = os.path.join(config.checkpoint[:-3], name + '.pt')
        config.optimization.input_variance = sigma ** 2
        config.optimization.loss_terms.empirical = empirical
        config.optimization.loss_terms.expectation = expectation
        return config


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


def check_experiments():
    experiments = glob('exps/**/emp_*', recursive=True)
    finished = set(os.path.dirname(f)
                   for f in glob('exps/**/epoch_*', recursive=True))
    unfinished = [exp for exp in experiments if exp not in finished]
    if len(unfinished) > 0:
        print('Some experiments didn\'t finish:')
        print('\n'.join(unfinished))

    else:
        print('All experiments has finished')

    print('#########################')

    for file in glob('exps/**/log.out', recursive=True):
        with open(file, 'r') as f:
            content = [line.strip() for line in f.readlines()]
            if len(content) > 0 and 'Achieved' not in content[-1]:
                index = 0
                for i in range(len(content) - 1, -1, -1):
                    if '/s]' in content[i]:
                        index = i
                        break
                print(file)
                print('\n'.join(content[index:]))
                print('.......................')


def main(device_index, multiple_gpus=True):
    datasets = ['mnist', 'cifar10', 'cifar100']
    emps = [0, 1]
    exps = [0, 0.5, 1.0, 1.5, 2]
    sigmas = [0, 0.1, 0.2, 0.3, 0.5, 0.75, 1]

    count = torch.cuda.device_count()
    experiments = list(product(datasets, emps, sigmas, exps))
    for i, (dataset, emp, sigma, exp) in enumerate(experiments):
        print(f'|#| Iteration {i + 1} out of {len(experiments)}...')
        if multiple_gpus and i % count != device_index:
            continue
        try:
            LeNetGNMTrainer.experiment(dataset, emp, sigma, exp,
                                       device=f'cuda:{device_index}')
        except Exception as exc:
            print(f'\n{str(exc)}')


if __name__ == '__main__':
    parser = ArgumentParser(description='PoC: Training with expectation.')
    parser.add_argument('-c', '--check', default=False,
                        action='store_true',
                        help='Check all run experiments.')
    parser.add_argument('-i', '--device-index', default=0, type=int,
                        help='The device index of the GPU.')
    parser.add_argument('-m', '--multiple-gpus', default=False,
                        action='store_true',
                        help='Should run on multiple GPUs.')
    args = parser.parse_args()
    if args.check:
        check_experiments()
    else:
        delattr(args, 'check')
        main(**vars(args))
