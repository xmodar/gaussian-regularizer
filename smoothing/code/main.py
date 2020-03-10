import datetime
import json
import os
from itertools import product
from pathlib import Path
from time import time

import torch

from architectures import get_architecture
from core import Smooth
from datasets import get_dataset

if __name__ == '__main__':
    index = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
    paths = []
    for p in reversed(sorted(Path('../results').glob('*/*'))):
        # if p.parent.name == 'ours' and int(p.name) < 86:
        #     continue
        paths.append(p)

    sigmas = [0.12, 0.25, 0.5, 1.0]
    path, sigma = list(product(paths, sigmas))[index]

    N0 = 100
    skip = 20
    N = 100000
    batch = 1000
    alpha = 0.001

    # load the base classifier
    with open(path / 'config.json') as f:
        config = json.load(f)
    checkpoint = torch.load(path / 'state_dict.pt')['model']
    base_classifier = get_architecture('cifar_' + config['model'], 'cifar10')
    if 'resnet' in config['model']:
        checkpoint = dict(
            zip(base_classifier[1].state_dict().keys(), checkpoint.values()))
    base_classifier[1].load_state_dict(checkpoint)

    # create the smoothed classifier g
    smoothed_classifier = Smooth(base_classifier, 10, sigma)

    # prepare output file
    with open(path / f'{sigma}_out.txt', 'w') as f:
        print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

        # iterate through the dataset
        for i, (x, label) in enumerate(get_dataset('cifar10', 'test')):

            # only certify every skip examples
            if i % skip != 0:
                continue

            print('Index: {}.'.format(i))

            before_time = time()
            # certify the prediction of g around x
            x = x.cuda()
            prediction, radius = smoothed_classifier.certify(
                x, N0, N, alpha, batch)
            after_time = time()
            correct = int(prediction == label)

            time_elapsed = str(
                datetime.timedelta(seconds=(after_time - before_time)))
            print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(i, label, prediction,
                                                     radius, correct,
                                                     time_elapsed),
                  file=f,
                  flush=True)
