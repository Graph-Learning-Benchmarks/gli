# Ogbn-arxiv
## Data Description

The **ogbn-arxiv** dataset is a directed graph, representing the citation network between all Computer Science (CS) arXiv papers indexed by MAG. Each node is an arXiv paper and each directed edge indicates that one paper cites another one. Each paper comes with a 128-dimensional feature vector obtained by averaging the embeddings of words in its title and abstract.

Statistics:
1. Nodes: 169343
2. Edges: 1166243


#### Citation
- Original Source
```
@inproceedings{Wu2018Stanford,
  title={Microsoft academic graph: When experts are not enough. },
  author={Kuansan Wang, Zhihong Shen, Chiyuan Huang, Chieh-Han Wu, Yuxiao Dong, and Anshul Kanakia},
  booktitle={Quantitative Science Studies},
  pages={396=413},
  year={2020}
}
```
- Current Version
```
@inproceedings{
  title={Distributed representationsof words and phrases and their compositionality},
  author={Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean},
  booktitle={In Advances in Neural Information Processing Systems (NeurIPS)},
  pages={3111=3119},
  year={2013}
}
```

## Available Tasks
### [OGB](https://ogb.stanford.edu/docs/nodeprop/)
- Task type: `NodeClassification`

#### Citation
```
@inproceedings{
  title={Distributed representationsof words and phrases and their compositionality},
  author={Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean},
  booktitle={In Advances in Neural Information Processing Systems (NeurIPS)},
  pages={3111=3119},
  year={2013}
}
```

## Preprocessing
The data files and task config file in GLB format are transformed from the OGB implementation. 

### Requirements
The preprocessing code requires the following package.
```
ogb >= 1.1.1
```