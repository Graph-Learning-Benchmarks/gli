# Chameleon

## Dataset Description

Chameleon is a page-page network on specific topics in Wikipedia. In this dataset, nodes represent web pages and edges are mutual links between pages. And node features correspond to several informative nouns in the Wikipedia pages. The nodes are classified into five categories.


Statistics:
- Nodes: 2277
- Edges: 36101
- Number of Classes: 5

#### Citation
- Original Source
  + [Website](https://github.com/benedekrozemberczki/datasets#wikipedia-article-networks)
  + LICENSE: [MIT](https://github.com/benedekrozemberczki/datasets/blob/master/LICENSE)
```
@article{rozemberczki2021multi,
  title={Multi-scale attributed node embedding},
  author={Rozemberczki, Benedek and Allen, Carl and Sarkar, Rik},
  journal={Journal of Complex Networks},
  volume={9},
  number={2},
  pages={cnab014},
  year={2021},
  publisher={Oxford University Press}
}
```
- Current Version
  + [Website](https://github.com/graphdml-uiuc-jlu/geom-gcn)
  + LICENSE: missing
```
@article{pei2020geom,
  title={Geom-gcn: Geometric graph convolutional networks},
  author={Pei, Hongbin and Wei, Bingzhe and Chang, Kevin Chen-Chuan and Lei, Yu and Yang, Bo},
  journal={arXiv preprint arXiv:2002.05287},
  year={2020}
}
```

## Available Tasks

### MUSAE

- Task type: `NodeClassification`

This is a node classification task with fixed split from [MUSAE](https://github.com/benedekrozemberczki/MUSAE).

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
The data files and task config file in GLI format are transformed from the [torch_geometric.datasets](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html). Check `chameleon.ipynb` for the preprocessing.


### Requirements

The preprocessing code requires the following packages.

```
numpy==1.22.3
scipy==1.7.3
torch==1.11.0
torch_geometric==2.0.4
```
