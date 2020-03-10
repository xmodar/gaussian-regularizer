import os
from itertools import product, repeat
from pathlib import Path
from pprint import pprint

from experiments import GaussianExp

if __name__ == '__main__':
    methods = {
        'nominal':
            dict(sigma=[0.0], alpha=[0.0], gamma=[0.0]),
        'cohen':
            dict(
                sigma=[0.12, 0.25, 0.5, 1.0],
                alpha=[0.0],  # no moment training
                gamma=[1.0],  # only noisy input
            ),
        'augment':
            dict(
                sigma=[0.12, 0.25, 0.5, 1.0],
                alpha=[0.0],  # no moment training
                gamma=[0.5],  # sometimes augment
            ),
        'ours':
            dict(
                sigma=[0.12, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
                alpha=[1.0, 0.9, 0.75, 0.5, 0.25, 0.12],
                gamma=[0.0],  # only clean input
            ),
    }

    experiments = []
    root_path = Path('./results')
    models = ['alexnet', 'vgg16_bn', 'resnet8', 'resnet56', 'resnet110']
    for name, method in methods.items():
        keys = repeat(('name', 'model') + tuple(method.keys()))
        values = product([name], models, *method.values())
        experiments += (dict(zip(*kv)) for kv in zip(keys, values))

    run = os.environ.get('SLURM_ARRAY_TASK_ID')
    if run:
        experiment = experiments[int(run)]
        config = GaussianExp.arg_parser().parse_args()
        config.log = str(root_path / experiment.pop('name'))
        config.epochs = 100
        vars(config).update(experiment)
        gaussian_exp = GaussianExp(**vars(config))
        pprint(gaussian_exp.config)
        gaussian_exp.train()
    else:
        pprint(experiments)
        print(f'Total number of experiments: {len(experiments)}')
