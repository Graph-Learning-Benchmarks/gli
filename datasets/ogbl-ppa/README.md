# Ogbl-ppa
## Data Description

 The **ogbl-ppa** dataset is an undirected, unweighted graph. Nodes represent proteins from 58 different species, and edges indicate biologically meaningful associations between proteins, e.g., physical interactions, co-expression, homology or genomic neighborhood

Statistics:
1. Nodes: 576,289
2. Edges: 30,326,273	


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
The data files and task config file in GLI format are transformed from the OGB implementation. 

### Requirements
The preprocessing code requires the following package.
```
ogb >= 1.1.1
```