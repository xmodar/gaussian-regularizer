# main source: https://click.palletsprojects.com/en/7.x/
# nice tutorial: https://www.youtube.com/watch?v=kNke39OZ2k0
import sys

import torch  # type: ignore
import click

from typing import Callable

PREFIX = 'CVPR'
FILE = 'args.txt'
DEBUG = True

__version__ = '0.1.0'
__all__ = ['main', 'all', 'show', 'dump', 'file', 'run']


class Config:

    def __init__(self, debug=DEBUG):
        self.debug = debug


pass_config = click.make_pass_decorator(Config, ensure=True)


@click.group()
@click.option(
    '-r/-d',
    '--run/--debug',
    'run_mode',
    is_flag=True,
    default=not DEBUG,
    show_default=True,
    envvar='_'.join([PREFIX, 'RUN']),  # get environment variable if defined
    help='Whether to run without debugging.')
@click.version_option(__version__, '-v', '--version', message='%(version)s')
@pass_config
def main(config, run_mode):
    # type: (Config, bool) -> None
    '''PoC: Training with expectation.'''
    config.debug = not run_mode
    if config.debug:
        print('Running in debugging mode...')


def gpu_args():
    # type: () -> Callable
    gpu = click.option(
        '-g',
        '--gpu',
        type=click.IntRange(min=-1, max=torch.cuda.device_count() - 1),
        default=0,
        show_default=True,
        envvar='_'.join([PREFIX, 'GPU']),
        help='The device index of the GPU (-1 to use CPU).')
    multiple = click.option(
        '-m/-s',
        '--multiple/--single',
        is_flag=True,
        default=False,
        show_default=True,
        envvar='_'.join([PREFIX, 'MULTIPLE']),
        help='Whether to run in multiple GPUs setup.')
    return lambda f: gpu(multiple(f))


@main.command()
@gpu_args()
@pass_config
def all(config, gpu, multiple):
    # type: (Config, int, bool) -> None
    '''Run the remaining experiments.'''
    mode = 'debug' if config.debug else 'run'
    print(f'{gpu} - {multiple} : {mode}')


@main.command()
@click.option(
    '-l/-f',
    '--little/--full',
    'short',
    is_flag=True,
    default=True,
    show_default=True,
    help='Whether to print stats summary or full list.')
@gpu_args()
def show(short, gpu, multiple):
    # type: (bool, int, bool) -> None
    '''Show the experiments.'''
    print(f'{short} - {gpu} - {multiple}')


@main.command()
@click.option(
    '-p',
    '--path',
    'file',
    type=click.Path(),
    default=FILE,
    show_default=True,
    help=f'Which file to dump the arguments list to.')
@gpu_args()
def dump(file, gpu, multiple):
    # type: (str, int, bool) -> None
    '''Output to file the remaining experiments.'''
    print(f'{file} - {gpu} - {multiple}')


@main.command()
@click.option(
    '-p',
    '--path',
    # consider using click.File() and using default='-' (stdin/stdout)
    type=click.Path(exists=True, dir_okay=False),
    default=FILE,
    show_default=True,
    help=f'Which file to read the arguments list from.')
@click.option(
    '-e',
    '--experiment',
    type=click.IntRange(min=0),
    help='The index to run from file.')
@gpu_args()
def file(path, experiment, gpu, multiple):
    # type: (str, int, int, bool) -> None
    '''Run experiments from file.'''
    print(f'{path} - {experiment} - {gpu} - {multiple}')


@main.command()
@click.option(
    '-m',
    '--model',
    required=True,
    default='lenet',
    show_default=True,
    type=click.Choice(['lenet', 'alexnet', 'vgg16']))  # list(Trainer.network)
@click.option(
    '-d',
    '--dataset',
    required=True,
    default='mnist',
    show_default=True,
    type=click.Choice(['mnist', 'cifar10',
                       'cifar100']))  # list(Trainer.dataset)
@click.option(
    '-emp',
    '--empirical',
    type=float,
    default=1,
    show_default=True,
    help='Coefficient of the empirical loss.')
@click.option(
    '-sig',
    '--sigma',
    type=click.FloatRange(min=0),  # type: ignore
    default=0,
    show_default=True,
    help='Input standard deviation.')
@click.option(
    '-aug',
    '--augmentation',
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help='Number of data augmentation folds.')
@click.option(
    '-exp',
    '--expectation',
    type=float,
    default=0,
    show_default=True,
    help='Coefficient of the expectation loss.')
@click.option(
    '-e',
    '--epochs',
    type=click.IntRange(min=0),
    default=100,
    show_default=True,
    help='Number of training epochs.')
@click.option(
    '-on',
    '--device',
    type=str,
    default='cuda',
    show_default=True,
    help='Which device to run the experiments on.')
@click.option(
    '-n',
    '--name',
    type=str,
    default='default',
    show_default=True,
    help='The name of the experiment.')
def run(model, dataset, empirical, sigma, augmentation, expectation, epochs,
        device, name):
    # type: (str, str, float, float, int, float, int, str, str) -> None
    '''Run a specific experiment.'''
    print(f'{model} - {dataset} - {empirical} - {sigma} '
          f'- {augmentation} - {expectation} - {epochs} - {device} - {name}')


if __name__ == "__main__":
    main(auto_envvar_prefix=PREFIX)
    # this line will not be reached as main() exists the program when done
