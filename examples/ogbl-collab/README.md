# OGBL-COLLAB

## Dataset Description
The ogbl-collab dataset is an undirected graph, representing a subset of the collaboration network between authors indexed by MAG[1]. Each node represents an author and edges indicate the collaboration between authors. All nodes come with 128-dimensional features, obtained by averaging the word embeddings of papers that are published by the authors. All edges are associated with two meta-information: the year and the edge weight, representing the number of co-authored papers published in that year. The graph can be viewed as a dynamic multi-graph since there can be multiple edges between two nodes if they collaborate in more than one year.


[1] Kuansan Wang, Zhihong Shen, Chiyuan Huang, Chieh-Han Wu, Yuxiao Dong, and Anshul Kanakia. Microsoft academic graph: When experts are not enough. Quantitative Science Studies, 1(1):396–413, 2020.


Statistics:
- Nodes: 235868
- Edges: 2358104

#### Citation

```
@article{wang2020microsoft,
  title={Microsoft academic graph: When experts are not enough},
  author={Wang, Kuansan and Shen, Zhihong and Huang, Chiyuan and Wu, Chieh-Han and Dong, Yuxiao and Kanakia, Anshul},
  journal={Quantitative Science Studies},
  pages={396--413},
  year={2020}
}
```

## Available Tasks

### Planetoid

- Task type: `TimeDependentLinkPrediction`


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

The data files and task config file in GLB format are transformed from the OGB implementation. 


### Requirements

The preprocessing code requires the following packages.

```
scipy==1.7.1
ogb==1.3.2
```
