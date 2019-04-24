# Gaussian Regularizer

Despite the impressive performance of deep neural networks (DNNs) on numerous vision tasks, they still exhibit yet-to-understand uncouth behaviours. One puzzling behaviour is the subtle sensitive reaction of DNNs to various noise attacks. Such a nuisance has strengthened the line of research around developing and training noise-robust networks. In this work, we propose a new training regularizer that aims to minimize the probabilistic expected training loss of a DNN subject to a generic Gaussian input. We provide an efficient and simple approach to approximate such a regularizer for arbitrary deep networks. This is done by leveraging the analytic expression of the output mean of a shallow neural network; avoiding the need for the memory and computationally expensive data augmentation. We conduct extensive experiments on LeNet and AlexNet on various datasets including MNIST, CIFAR10, and CIFAR100 demonstrating the effectiveness of our proposed regularizer. In particular, we show that networks that are trained with the proposed regularizer benefit from a boost in robustness equivalent to performing 3-21 folds of data augmentation.

### Installation

All the requirements are listed in [requirements.txt](./requirements.txt).

```shell
pip install -r requirements.txt
python main.py --help
```

The training logs for the experiments reported in the paper are all published under releases as a single zip file.

### Cite

This is the official implementation of the method described in [this paper](https://arxiv.org/abs/1904.11005):

```bibtex
@misc{alfadly2019analytical,
    title={Analytical Moment Regularizer for Gaussian Robust Networks},
    author={Modar Alfadly and Adel Bibi and Bernard Ghanem},
    year={2019},
    eprint={1904.11005},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

### License

MIT

### Author

[Modar M. Alfadly](https://modar.me/)

### Contributors

I would gladly accept any pull request that improves any aspect of this repository.