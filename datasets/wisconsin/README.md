# Wisconsin

## Dataset Description

WebKB is a webpage dataset collected from computer science departments of various universities by Carnegie Mellon University. This is the subdataset of it, Wisconsin, where nodes represent web pages, and edges are hyperlinks between them. Node features are the bag-of-words representation of web pages. The web pages are manually classified into the five categories, student, project, course, staff, and faculty.

Statistics:
- Nodes: 251
- Edges: 515
- Number of Classes: 5

#### Citation
- Original Source
  + [Website](https://www.cs.cmu.edu/~webkb/)
  + LICENSE: missing
```
@inproceedings{craven1998learning,
  title={Learning to Extract Symbolic Knowledge from the World Wide Web},
  author={CRAVEN, M},
  booktitle={Proc. of the 15th National Conference on Artificial Intelligence},
  pages={509--516},
  year={1998},
  organization={AAAI Press}
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
- Previous Version
  + [Website](https://linqs.org/datasets/#webkb)
  + LICENSE: missing
```
@conference{lu:icml03,
    title = {Link-Based Classification},
    author = {Qing Lu and Lise Getoor},
    booktitle = {International Conference on Machine Learning},
    year = {2003},
    _publisher = {HP},
    address = {Washington, DC, USA},
}
```

## Available Tasks

### Webkb

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
The data files and task config file in GLI format are transformed from the [torch_geometric.datasets](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html). Check `wisconsin.ipynb` for the preprocessing.


### Requirements

The preprocessing code requires the following packages.

```
numpy==1.22.3
scipy==1.7.3
torch==1.11.0
torch_geometric==2.0.4
```
