# OGBL-COLLAB

## Dataset Description
The ogbl-collab dataset is an undirected graph, representing a subset of the collaboration network between authors indexed by MAG[1]. Each node represents an author and edges indicate the collaboration between authors. All nodes come with 128-dimensional features, obtained by averaging the word embeddings of papers that are published by the authors. All edges are associated with two meta-information: the year and the edge weight, representing the number of co-authored papers published in that year. The graph can be viewed as a dynamic multi-graph since there can be multiple edges between two nodes if they collaborate in more than one year.


[1] Kuansan Wang, Zhihong Shen, Chiyuan Huang, Chieh-Han Wu, Yuxiao Dong, and Anshul Kanakia. Microsoft academic graph: When experts are not enough. Quantitative Science Studies, 1(1):396–413, 2020.


Statistics:
- Nodes: 235868
- Edges: 2358104

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
  - LICENSE: [ODC-BY](https://ogb.stanford.edu/docs/linkprop/)
```
@article{hu2022stanford,
  title={Open Graph Benchmark: Datasets for Machine Learning on Graphs},
  author={Hu, Weihua and Fey, Matthias and Zitnik, Marinka and Dong, Yuxiao and Ren, Hongyu and Liu, Bowen and Catasta, Michele and Leskovec, Jure},
  year={2021}
}
```
## Available Tasks

### OGB

- Task type: `TimeDependentLinkPrediction`


#### Citation

```
@article{hu2022stanford,
  title={Open Graph Benchmark: Datasets for Machine Learning on Graphs},
  author={Hu, Weihua and Fey, Matthias and Zitnik, Marinka and Dong, Yuxiao and Ren, Hongyu and Liu, Bowen and Catasta, Michele and Leskovec, Jure},
  year={2021}
}
```

## Preprocessing

Data file(s) is transformed from [OGB](https://ogb.stanford.edu/). Check ogbl-collab.ipynb for preprocessing.


### Requirements

The preprocessing code requires the following packages.

```
scipy==1.7.1
ogb==1.3.2
numpy
torch
```
