# MNIST
## Data Description

*MNIST* database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. The database is also widely used for training and testing in the field of machine learning. It was created by "re-mixing" the samples from NIST's original datasets

Statistics:
- Nodes: 4939668
- Edges: 39517344

#### Citation
- Original Source
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
