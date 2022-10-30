# MNIST
## Data Description

*MNIST* is a large database of handwritten digits that is commonly used for training various image processing systems. The database is also widely used for training and testing in the field of machine learning. In this dataset, the original images are converted to graphs using superpixels, which represent small regions of homogeneous intensity in images. For each image, the nodes are superpixels and the graph is a k-nearest neighbor graph on the superpixels.

Statistics:
- Nodes: 4939668
- Edges: 39517344

#### Citation
- Original Source
  + [Website](http://yann.lecun.com/exdb/mnist/)
  + LICENSE: missing
```
@ARTICLE{726791,
  author={Lecun, Y. and Bottou, L. and Bengio, Y. and Haffner, P.},
  journal={Proceedings of the IEEE},
  title={Gradient-based learning applied to document recognition},
  year={1998},
  volume={86},
  number={11},
  pages={2278-2324},
  doi={10.1109/5.726791}
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
- Previous Version
  + [Website](https://github.com/bknyaz/graph_attention_pool/tree/master/data)
  + LICENSE: [ECL-2.0](https://github.com/bknyaz/graph_attention_pool/blob/master/LICENSE.md)
```
@article{knyazev2019understanding,
  title={Understanding attention and generalization in graph neural networks},
  author={Knyazev, Boris and Taylor, Graham W and Amer, Mohamed},
  journal={Advances in neural information processing systems},
  volume={32},
  year={2019}
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
