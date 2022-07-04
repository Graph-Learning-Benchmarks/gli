# Wisconsin

## Dataset Description

WebKB is a webpage dataset collected from computer science departments of various universities by Carnegie Mellon University. This is the subdataset of it, Wisconsin, where nodes represent web pages, and edges are hyperlinks between them. Node features are the bag-of-words representation of web pages. The web pages are manually classified into the five categories, student, project, course, staff, and faculty.

Statistics:
- Nodes: 251
- Edges: 515
- Number of Classes: 5

#### Citation

```
@online{webkb,
  author={WebKb Group},
  title={CMU World Wide Knowledge Base},
  date={2001-01},
  url={http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-11/www/wwkb/}
}
```

## Available Tasks

### Webkb

- Task type: `NodeClassification`

This is a node classification task with fixed split from [Webkb](https://github.com/kimiyoung/planetoid).

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
The data files and task config file in GLB format are transformed from the [torch_geometric.datasets](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html). Check `wisconsin.ipynb` for the preprocessing.


### Requirements

The preprocessing code requires the following packages.

```
numpy==1.22.3
scipy==1.7.3
torch==1.11.0
torch_geometric==2.0.4
```
