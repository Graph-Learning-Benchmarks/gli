# CIFAR
## Data Description

*CIFAR*  (Canadian Institute For Advanced Research) is a collection of images that are commonly used to train machine learning and computer vision algorithms. It is one of the most widely used datasets for machine learning research. The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes.

Statistics:

- Nodes: 7058005
- Edges: 56464040

#### Citation
- Original Source
```
@article{krizhevsky2009learning,
  added-at = {2021-01-21T03:01:11.000+0100},
  author = {Krizhevsky, Alex},
  biburl = {https://www.bibsonomy.org/bibtex/2fe5248afe57647d9c85c50a98a12145c/s364315},
  interhash = {cc2d42f2b7ef6a4e76e47d1a50c8cd86},
  intrahash = {fe5248afe57647d9c85c50a98a12145c},
  keywords = {},
  pages = {32--33},
  timestamp = {2021-01-21T03:01:11.000+0100},
  title = {Learning Multiple Layers of Features from Tiny Images},
  url = {https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf},
  year = 2009
}
```
- Current Version
#### Citation
```
@misc{https://doi.org/10.48550/arxiv.2003.00982,
  doi = {10.48550/ARXIV.2003.00982},
  url = {https://arxiv.org/abs/2003.00982},
  author = {Dwivedi, Vijay Prakash and Joshi, Chaitanya K. and Luu, Anh Tuan and Laurent, Thomas and Bengio, Yoshua and Bresson, Xavier},
  keywords = {Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Benchmarking Graph Neural Networks},
  publisher = {arXiv},
  year = {2020},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```
## Available Tasks
### [Benchmarking-gnns](https://github.com/graphdeeplearning/benchmarking-gnns/)
- Task type:  `GraphClassification`

#### Citation
```
@misc{https://doi.org/10.48550/arxiv.2003.00982,
  doi = {10.48550/ARXIV.2003.00982},
  url = {https://arxiv.org/abs/2003.00982},
  author = {Dwivedi, Vijay Prakash and Joshi, Chaitanya K. and Luu, Anh Tuan and Laurent, Thomas and Bengio, Yoshua and Bresson, Xavier},
  keywords = {Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Benchmarking Graph Neural Networks},
  publisher = {arXiv},
  year = {2020},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```

## Preprocessing
The data files and task config file in GLI format are transformed from the DGL implementation.

### Requirements
The preprocessing code requires the following package.
```
numpy >= 1.21.5
```
