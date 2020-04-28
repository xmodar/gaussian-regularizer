import os
from itertools import product, repeat
from pathlib import Path
from pprint import pprint

from evaluate import get_robustness
from experiments import GaussianExp

if __name__ == '__main__':
    methods = {
        'consistency':
            dict(
                moment_loss=['consistency'],
                sigma=[0.12, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
                alpha=[1.0, 0.75, 0.5, 0.25],
                gamma=[0.5],
            ),
        'cross_entropy_outputs':
            dict(
                moment_loss=['cross_entropy_outputs'],
                sigma=[0.12, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
                alpha=[1.0, 0.75, 0.5, 0.25],
                gamma=[0.5],
            ),
        'cross_entropy':
            dict(
                moment_loss=['cross_entropy'],
                sigma=[0.12, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
                alpha=[1.0, 0.75, 0.5, 0.25],
                gamma=[0.5],
            ),
    }

    experiments = []
    root_path = Path('./fixes')
    models = ['resnet56']
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
        get_robustness(gaussian_exp.config['path'])
    else:
        pprint(experiments)
        print(f'Total number of experiments: {len(experiments)}')
