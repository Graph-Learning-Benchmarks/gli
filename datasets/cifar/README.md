# CIFAR
## Data Description

*CIFAR* is a collection of images that are commonly used to train machine learning and computer vision algorithms. It is one of the most widely used datasets for machine learning research. The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes. In this dataset, the original images are converted to graphs using superpixels, which represent small regions of homogeneous intensity in images. For each image, the nodes are superpixels and the graph is a k-nearest neighbor graph on the superpixels.

Statistics:

- Nodes: 7058005
- Edges: 56464040

#### Citation
- Original Source
  + [Website](https://www.cs.toronto.edu/~kriz/cifar.html)
  + LICENSE: missing
```
@Techreport{Krizhevsky_2009_17719,
  author = {Krizhevsky, Alex and Hinton, Geoffrey},
 address = {Toronto, Ontario},
 institution = {University of Toronto},
 number = {0},
 publisher = {Technical report, University of Toronto},
 title = {Learning multiple layers of features from tiny images},
 year = {2009},
 title_with_no_special_chars = {Learning multiple layers of features from tiny images}
}
```
- Current Version
  + [Website](https://github.com/graphdeeplearning/benchmarking-gnns/tree/master/data/superpixels)
  + LICENSE: [MIT](https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/LICENSE)
```
@article{dwivedi2020benchmarking,
  title={Benchmarking graph neural networks},
  author={Dwivedi, Vijay Prakash and Joshi, Chaitanya K and Laurent, Thomas and Bengio, Yoshua and Bresson, Xavier},
  journal={arXiv preprint arXiv:2003.00982},
  year={2020}
}
```

## Available Tasks
### [Benchmarking-gnns](https://github.com/graphdeeplearning/benchmarking-gnns/)
- Task type:  `GraphClassification`

#### Citation
```
@article{dwivedi2020benchmarking,
  title={Benchmarking graph neural networks},
  author={Dwivedi, Vijay Prakash and Joshi, Chaitanya K and Laurent, Thomas and Bengio, Yoshua and Bresson, Xavier},
  journal={arXiv preprint arXiv:2003.00982},
  year={2020}
}
```

## Preprocessing
The data files and task config file in GLI format are transformed from the [benchmarking-gnns repository](https://github.com/graphdeeplearning/benchmarking-gnns/).

### Requirements
The preprocessing code requires the following package.

```
numpy >= 1.21.5
scikit-learn >= 0.24.2
scipy >= 1.5.4
torch >= 1.10.2
```
