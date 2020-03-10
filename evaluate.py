import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from experiments import GaussianExp
# from smoothing.code.analyze import  plot_certified_accuracy
from smoothing.code.analyze import ApproximateAccuracy, Line


def ring(size, sigma=1, tolerance=0, batch=None, dtype=None, device=None):
    if batch is not None:
        size = [batch] + list(size)
    noise = torch.randn(size, dtype=dtype, device=device)
    flat = noise.view(1 if batch is None else batch, -1)
    norms = flat.norm(dim=1, keepdim=True)
    if tolerance != 0:
        sigma = torch.rand_like(norms) * (2 * tolerance) + (sigma - tolerance)
    flat.mul_(sigma * flat.size(1)**0.5 / norms)
    return noise


@torch.no_grad()
def gaussian_robustness(model, loader, device, sigmas_range=(0, 0.5, 30)):
    model.to(device).eval()

    # compute the output predictions for the clean images
    count = 0
    clean_labels = []
    for images, _ in loader:
        labels = model.forward(images.to(device)).argmax(dim=-1)
        clean_labels.append(labels)
        count += images.size(0)

    # get the noise levels (sigmas)
    std = loader.dataset.transform.transforms[-1].std
    input_range = sum(float(1 / s) for s in std) / len(std)
    sigmas = torch.linspace(*sigmas_range, device=device)
    accuracies = torch.zeros(sigmas.numel(), device=device)
    # pylint: disable=undefined-loop-variable
    kwargs = {
        'device': device,
        'dtype': images.dtype,
        'size': images[0, ...].size(),
        'tolerance': float((sigmas[1] - sigmas[0]) / 2),
    }
    # pylint: enable=undefined-loop-variable

    for i, sigma in enumerate(tqdm(sigmas)):
        sigma = float(sigma * input_range)
        for (images, _), labels in zip(loader, clean_labels):
            noise = ring(batch=images.size(0), sigma=sigma, **kwargs)
            out = model.forward(noise.add_(images.to(device)))
            accuracies[i] += int(out.argmax(dim=-1).eq(labels).sum())

    accuracies /= count
    robustness = torch.trapz(accuracies, x=sigmas) / (sigmas[-1] - sigmas[0])
    return robustness, (sigmas, accuracies)


@torch.no_grad()
def sigma_robustness(model, loader, device, sigma, samples=30):
    model.to(device).eval()

    # compute the output predictions for the clean images
    count = 0
    clean_labels = []
    for images, _ in loader:
        labels = model(images.to(device)).argmax(dim=-1)
        clean_labels.append(labels)
        count += images.size(0)

    # get the noise levels (sigmas)
    std = loader.dataset.transform.transforms[-1].std
    input_range = sum(float(1 / s) for s in std) / len(std)
    sigma = float(sigma * input_range)

    robustness = 0
    for (images, _), labels in zip(tqdm(loader), clean_labels):
        votes = 0
        for _ in range(samples):
            n = torch.randn_like(images, device=device).mul_(sigma)
            votes += model(n.add_(images.to(device))).argmax(dim=-1) == labels
        robustness += (votes > samples / 2).sum().item()
    robustness /= count
    return robustness


# %matplotlib inline


def json_load(path):
    with open(path, 'r') as f:
        return json.load(f)


def json_save(data, path):

    with open(path, 'w') as f:
        json.dump(data, f)
    return data


def get_config(path):
    return json_load(Path(path) / 'config.json')


def get_metrics(path):
    return json_load(Path(path) / 'metrics.json')


def get_experiment(path):
    exp = GaussianExp(**get_config(path))
    exp.load_state_dict(torch.load(Path(path) / 'state_dict.pt'))
    return exp


def get_robustness(path):
    robust_path = Path(path) / 'robustness.json'
    if robust_path.exists():
        return json_load(robust_path)
    exp = get_experiment(path)
    r, (xs, ys) = gaussian_robustness(exp.model, exp.data['test'], exp.device)
    robusness = {
        'AuC': float(r),
        'sigmas': xs.tolist(),
        'accuracies': ys.tolist(),
    }
    return json_save(robusness, robust_path)


def get_sigma_robustness(path, sigma=None, rounded=3):
    if sigma:
        sigma = str(round(float(sigma), rounded))
    robust_path = Path(path) / f'sigma_robustness.json'
    sr = {}
    if robust_path.exists():
        sr = json_load(robust_path)
        if sigma in sr or sigma is None:
            return sr
    e = get_experiment(path)
    sr[sigma] = sigma_robustness(e.model, e.data['test'], e.device,
                                 float(sigma))
    json_save(sr, robust_path)
    return {float(k): v for k, v in sr.items()}


def get_sigma_certificates(path):
    out = {}
    for txt in list(Path(path).glob('*.txt')):
        sigma = float(txt.name.replace('_out.txt', ''))
        a = ApproximateAccuracy(txt)
        xs = np.linspace(0, 1.5, 100)
        auc = np.trapz(a.at_radii(xs), xs)
        out[sigma] = round(float(auc), 4)
    return out


def get_title(path):
    config = get_config(path)
    method = Path(config['log']).name.title()
    model = config['model'].title()
    s = config['sigma']
    a = config['alpha']
    if s == 0:
        title = f'Nominal {model}'
    else:
        title = f'{method} {model} [$\\sigma={s:.2f}, \\alpha={a:.2f}$]'
    metrics = get_metrics(path)['test']
    accuracy = metrics['accuracy'] * 100
    title += f'\nAccuracy={accuracy:.2f}%'
    if 'moment_accuracy' in metrics:
        moment_accuracy = metrics['moment_accuracy'] * 100
        title += f'\nMoment Accuracy={moment_accuracy:.2f}%'
    robustness = get_robustness(path)['AuC'] * 100
    title += f'\nRobustness={robustness:.2f}%'
    return title


def get_all_results(root_path):
    results = defaultdict(list)
    for path in sorted(Path(root_path).glob('**/state_dict.pt')):
        path = path.parent
        config = get_config(path)
        sigma = config['sigma']
        method = path.parent.name.title()
        results['method'] += [method]
        results['sigma'] += [sigma]
        results['model'] += [config['model']]
        results['alpha'] += [config['alpha']]
        results['gamma'] += [config['gamma']]
        metrics = get_metrics(path)['test']
        results['accuracy'] += [metrics['accuracy']]
        results['moment_accuracy'] += [metrics.get('moment_accuracy')]
        results['robustness'] += [get_robustness(path)['AuC']]
        results['sigma_robustness'] += [get_sigma_robustness(path)]
        results['sigma_certificates'] += [get_sigma_certificates(path)]
        results['path'] += [path]
    return pd.DataFrame(results)


def plot_certified_accuracy(title, lines, max_radius=1.5, radius_step=0.01):
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    for line in lines:
        plt.plot(
            radii * line.scale_x,
            line.quantity.at_radii(radii),
            line.plot_fmt,
        )
    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel('radius', fontsize=16)
    plt.ylabel('certified accuracy', fontsize=16)
    plt.legend(
        [method.legend for method in lines],
        loc='upper right',
        fontsize=16,
    )
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.show()


def plot_single_model(path):
    path = Path(path)
    suffix = '_out.txt'
    sigmas = sorted([
        float(f.name[:-len(suffix)])
        for f in path.iterdir()
        if f.name.endswith(suffix)
    ])
    labels = [f'$\\sigma={s:.2f}$' for s in sigmas]
    values = [ApproximateAccuracy(path / f'{s}_out.txt') for s in sigmas]
    lines = [Line(v, l) for v, l in zip(values, labels)]
    plot_certified_accuracy(get_title(path), lines)


def plot_best_per_sigma(df, title, criteria='accuracy'):
    labels, values = [], []
    for sigma in sorted(df.sigma.unique()):
        best = df.loc[df[df['sigma'] == sigma][criteria].idxmax()]
        if not (best['path'] / f'{sigma}_out.txt').exists():
            continue
        a, m = best.get('accuracy', 0), best.get('moment_accuracy', 0)
        labels += [f'$\\sigma={sigma:.2f}$ @ {a * 100:.2f}% x {m * 100:.2f}%']
        values += [ApproximateAccuracy(best['path'] / f'{sigma}_out.txt')]
    lines = [Line(v, l) for v, l in zip(values, labels)]
    plot_certified_accuracy(title, lines)


def plot_robustness(path):
    r = get_robustness(path)
    plt.plot(r['sigmas'], r['accuracies'])
    plt.title(f'Gaussian Robustness = {r[""] * 100:.02f}%')
    plt.ylim((0, 1.1))
    plt.show()
