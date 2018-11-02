import torch
import pandas as pd
from main import ARGS
from main import Trainer
from itertools import product
import matplotlib.pyplot as plt
from collections import namedtuple, OrderedDict


__all__ = ['results', 'name' 'print_results', 'aug_vs_exp', 'plot_dataframe']


def name(net, dataset, emp, sig, aug, exp):  # pylint: disable=W0613
    gauss = f'exp={exp:05.2f}' if aug == 0 else f'aug={aug:02d}   '
    return f'{net}-{dataset} [sig={sig:.2f} {gauss}]'


def print_results(result=None, plot=False):
    if isinstance(result, dict):
        res = [result]
    for res in result:
        r = res['summary'].test
        msg = name(res['model'], res['dataset'], res['emp'],
                   res['sig'], res['aug'], res['exp'])
        label = f'{msg}: acc={r.accuracy*100:.2f}% rob={r.robustness*100:.2f}%'
        if plot:
            plot = r.robustness_plot
            label = label[label.find('['):]
            plt.plot(plot.sigmas, plot.accuracies, label=label)
        else:
            print(label)
    if plot and result:
        plt.legend(prop={'family': 'monospace'}, bbox_to_anchor=(1, 1))
        plt.show()


def results(verbose=False, plot=False, **kwargs):
    all_results = OrderedDict((k[:-1], v) for k, v in ARGS.items())
    for k in kwargs:
        if k not in all_results:
            raise TypeError(f'Unexpected {k} not in {list(all_results)}.')
    for k in all_results:
        if k not in kwargs:
            continue
        function = isinstance(kwargs[k], type(results))
        iterable = any(isinstance(kwargs[k], t) for t in [list, tuple, set])
        if not function:
            values = kwargs[k] if iterable else [kwargs[k]]
            kwargs[k] = lambda x: x in values
        all_results[k] = list(filter(kwargs[k], all_results[k]))

    out = []
    for args in product(*all_results.values()):
        config = Trainer.config(*args, name='poc')
        if config is None:
            continue
        try:
            out.append(Trainer.Results.results(config.log_dir))
        except Exception as exce:
            print(name(*args) + ': ', exce)
    if verbose or plot:
        print_results(out, plot=plot)
    return out


def plot_dataframe(df, title=None, table=False,
                   style=None, dot_style='o', line_style='+-'):
    dark = plt.rcParams['axes.facecolor'] == 'black'
    if style is None:
        style = ['D'] + [dot_style if sum(v == v) <= 1 else line_style
                         for v in df.values.transpose()[1:]]
    df.plot(style=style, title=title)
    plt.legend(prop={'family': 'monospace'}, loc='center left',
               bbox_to_anchor=(1.0, 0.5))
    if table:
        c = 'k' if dark else 'w'
        table = plt.table(
            cellText=df.values.transpose(),
            rowColours=[c] * len(df.columns),
            colColours=[c] * len(df.index),
            cellColours=[[c] * len(df.index)] * len(df.columns),
            rowLabels=df.columns,
            colLabels=df.index,
            bbox=[0, -0.65, 1, 0.6],
        )
        for cell in table.get_celld().values():
            cell.set_edgecolor('w' if dark else 'k')
    plt.show()


def aug_vs_exp(model, dataset, verbose=False):
    res = namedtuple('res', ('aug', 'sig', 'acc', 'rob'))
    i = 0
    best = {}
    best_value = {}
    results = []
    for result in Trainer.Results.all_results():
        if not result['model'].startswith(model):
            continue
        if result['dataset'] != dataset:
            continue
        if result['aug'] != 0 and result['sig'] > 1:
            continue
        # if result['exp'] != 0 and result['emp'] == 0:
        #     continue
        aug = result['aug']
        sig = result['sig']
        acc = result['summary'].test.accuracy
        rob = result['summary'].test.robustness
        if aug == 0:
            if sig not in best or best_value[sig] < acc:
                best[sig] = i
                best_value[sig] = acc
        results.append(res(aug, sig, acc, rob))
        i += 1
    if not results:
        return

    augs = sorted(set(a.aug for a in results))
    sigs = sorted(set(a.sig for a in results))

    def extract(attr):
        out = torch.zeros(len(augs), len(sigs)) / 0
        for i, result in enumerate(results):
            value = getattr(result, attr)
            if result.aug == 0 and i != best[result.sig]:
                continue
            index = (augs.index(result.aug), sigs.index(result.sig))
            out[index] = value
        df = pd.DataFrame(out.numpy(),
                          columns=[f'sig={v}' for v in sigs],
                          index=[f'aug={v:2d}' for v in augs])
        return df

    acc = extract('acc')
    rob = extract('rob')
    if verbose:
        plot_dataframe(acc, f'Test accuracy ({model}-{dataset}):')
        plot_dataframe(rob, f'Robustness ({model}-{dataset}):')
    return acc, rob
