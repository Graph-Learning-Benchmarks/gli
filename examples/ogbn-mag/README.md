# OGBL-MAG

## Dataset Description
The ogbn-mag dataset is a heterogeneous network composed of a subset of the Microsoft Academic Graph (MAG). It contains four types of entities—papers (736,389 nodes), authors (1,134,649 nodes), institutions (8,740 nodes), and fields of study (59,965 nodes)—as well as four types of directed relations connecting two types of entities—an author is “affiliated with” an institution, an author “writes” a paper, a paper “cites” a paper, and a paper “has a topic of” a field of study. Similar to ogbn-arxiv, each paper is associated with a 128-dimensional word2vec feature vector, and all the other types of entities are not associated with input node features.


Statistics:
- Nodes: 
  - papers (736,389 nodes) 
  - authors (1,134,649 nodes)
  - institutions (8,740 nodes)
  - fields of study (59,965 nodes)

#### Citation

```
@inproceedings{wang2020microsoft,
    title={Microsoft academic graph: When experts are not enough},
    author={Wang, Kuansan and Shen, Zhihong and Huang, Chiyuan and Wu, Chieh-Han and Dong, Yuxiao and Kanakia, Anshul},
    booktitle={Quantitative Science Studies},
    pages={396--413},
    year={2020}
}
```

## Available Tasks

### Planetoid

- Task type: `NodeClassification`


#### Citation

```
@inproceedings{wang2020microsoft,
    title={Microsoft academic graph: When experts are not enough},
    author={Wang, Kuansan and Shen, Zhihong and Huang, Chiyuan and Wu, Chieh-Han and Dong, Yuxiao and Kanakia, Anshul},
    booktitle={Quantitative Science Studies},
    pages={396--413},
    year={2020}
}
```

## Preprocessing

Check ogbg-mag.ipynb for preprocessing

### Requirements

The preprocessing code requires the following packages.

```
scipy==1.7.1
ogb==1.3.2
```
