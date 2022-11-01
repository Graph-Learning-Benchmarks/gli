# OGBN-MAG

## Dataset Description
The ogbn-mag dataset is a heterogeneous network composed of a subset of the Microsoft Academic Graph (MAG). It contains four types of entities—papers (736,389 nodes), authors (1,134,649 nodes), institutions (8,740 nodes), and fields of study (59,965 nodes)—as well as four types of directed relations connecting two types of entities—an author is “affiliated with” an institution, an author “writes” a paper, a paper “cites” a paper, and a paper “has a topic of” a field of study. Each paper is associated with a 128-dimensional word2vec feature vector, and all the other types of entities are not associated with input node features.


Statistics:
- Nodes:
  - papers (736,389 nodes)
  - authors (1,134,649 nodes)
  - institutions (8,740 nodes)
  - fields of study (59,965 nodes)

#### Citation
- Original Version
  - [Website](https://direct.mit.edu/qss/article/1/1/396/15572/Microsoft-Academic-Graph-When-experts-are-not)
  - LICENSE: Missing
```
@inproceedings{wang2020microsoft,
    title={Microsoft academic graph: When experts are not enough},
    author={Wang, Kuansan and Shen, Zhihong and Huang, Chiyuan and Wu, Chieh-Han and Dong, Yuxiao and Kanakia, Anshul},
    booktitle={Quantitative Science Studies},
    pages={396--413},
    year={2020}
}
```
- Current Version
  - [Website](https://ogb.stanford.edu/docs/linkprop/)
  - LICENSE: [ODC-BY](https://ogb.stanford.edu/docs/nodeprop/)
```
@article{hu2022stanford,
  title={Open Graph Benchmark: Datasets for Machine Learning on Graphs},
  author={Hu, Weihua and Fey, Matthias and Zitnik, Marinka and Dong, Yuxiao and Ren, Hongyu and Liu, Bowen and Catasta, Michele and Leskovec, Jure},
  year={2021}
}
```
## Available Tasks

### OGB

- Task type: `NodeClassification`


#### Citation
```
@article{hu2022stanford,
  title={Open Graph Benchmark: Datasets for Machine Learning on Graphs},
  author={Hu, Weihua and Fey, Matthias and Zitnik, Marinka and Dong, Yuxiao and Ren, Hongyu and Liu, Bowen and Catasta, Michele and Leskovec, Jure},
  year={2021}
}
```

## Preprocessing

Data file(s) is transformed from [OGB](https://ogb.stanford.edu/). Check ogbn-mag.ipynb for preprocessing.

### Requirements

The preprocessing code requires the following packages.

```
scipy==1.7.1
ogb==1.3.2
```
