# CORA

## Dataset Description

The CORA dataset contains a citation network with with documents as nodes and citations as edges. Each node has bag-of-words features of the document and a class label represents the research area this document belongs to.

Statistics:
- Nodes: 2708
- Edges: 10556
- Number of Classes: 7

#### Citation
- Original Source
  + [Website](https://linqs.org/datasets/#cora)
  + LICENSE: missing
```
@article{mccallum2000automating,
  title={Automating the construction of internet portals with machine learning},
  author={McCallum, Andrew Kachites and Nigam, Kamal and Rennie, Jason and Seymore, Kristie},
  journal={Information Retrieval},
  volume={3},
  number={2},
  pages={127--163},
  year={2000},
  publisher={Springer}
}
```
```
@inproceedings{lu:icml03,
    title = {Link-Based Classification},
    author = {Qing Lu and Lise Getoor},
    booktitle = {International Conference on Machine Learning (ICML)},
    year = {2003},
    _publisher = {HP},
    address = {Washington, DC, USA},
}
```

- Current Version
  + [Website](https://github.com/kimiyoung/planetoid)
  + LICENSE: [MIT](https://github.com/kimiyoung/planetoid/blob/master/LICENSE)
```
@inproceedings{yang2016revisiting,
  title={Revisiting semi-supervised learning with graph embeddings},
  author={Yang, Zhilin and Cohen, William and Salakhudinov, Ruslan},
  booktitle={International conference on machine learning},
  pages={40--48},
  year={2016},
  organization={PMLR}
}
```

- Previous Version
  + [Website](https://linqs.org/datasets/#cora)
  + LICENSE: missing
```
@article{sen2008collective,
  title={Collective classification in network data},
  author={Sen, Prithviraj and Namata, Galileo and Bilgic, Mustafa and Getoor, Lise and Galligher, Brian and Eliassi-Rad, Tina},
  journal={AI magazine},
  volume={29},
  number={3},
  pages={93--93},
  year={2008}
}
```

## Available Tasks

### Planetoid

- Task type: `NodeClassification`

This is a node classification task with fixed split from [planetoid](https://github.com/kimiyoung/planetoid).

#### Citation

```
@inproceedings{yang2016revisiting,
  title={Revisiting semi-supervised learning with graph embeddings},
  author={Yang, Zhilin and Cohen, William and Salakhudinov, Ruslan},
  booktitle={International conference on machine learning},
  pages={40--48},
  year={2016},
  organization={PMLR}
}
```

## Preprocessing

The data files and task config file in GLI format are transformed from the [DGL](https://www.dgl.ai) implementation. Check `cora.ipynb` for the preprocessing.


### Requirements

The preprocessing code requires the following packages.

```
scipy==1.7.1
dgl-cuda11.3==0.7.2
```
