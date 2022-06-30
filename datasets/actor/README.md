# Actor

## Dataset Description
This dataset is the actor-only induced subgraph of the film-director-actor-writer network (Tang et al., 2009). Each nodes correspond to an actor, and the edge between two nodes denotes co-occurrence on the same Wikipedia page. Node features correspond to some keywords in the Wikipedia pages. We classify the nodes into five categories in term of words of actor’s Wikipedia.


Statistics:
- Nodes: 7600
- Edges: 30019
- Number of Classes: 5

#### Citation

```
@inproceedings{tang2009social,
  title={Social influence analysis in large-scale networks},
  author={Tang, Jie and Sun, Jimeng and Wang, Chi and Yang, Zi},
  booktitle={Proceedings of the 15th ACM SIGKDD international conference on Knowledge discovery and mining},
  pages={807--816},
  year={2009}
  }
```
## Available Tasks

### Actor

- Task type: `NodeClassification`


#### Citation

```
@article{pei2020geom,
  title={Geom-gcn: Geometric graph convolutional networks},
  author={Pei, Hongbin and Wei, Bingzhe and Chang, Kevin Chen-Chuan and Lei, Yu and Yang, Bo},
  journal={arXiv preprint arXiv:2002.05287},
  year={2020}
}
```

## Preprocessing
The data files and task config file in GLB format are transformed from the [torch_geometric.datasets](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html). Check `actor.ipynb` for the preprocessing.


### Requirements

The preprocessing code requires the following packages.

```
numpy==1.22.3
scipy==1.7.3
torch==1.11.0
torch_geometric==2.0.4
```
