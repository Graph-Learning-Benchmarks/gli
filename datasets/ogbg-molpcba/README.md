# Ogbg-molpcba
## Data Description
**Ogbg-molpcba** is adopted from the MoleculeNet. It is a medium-sized molecular property prediction dataset. All the molecules are pre-processed using RDKit. Each graph represents a molecule, where nodes are atoms, and edges are chemical bonds. Input node features are 9-dimensional, containing atomic number and chirality, as well as other additional atom features such as formal charge and whether the atom is in the ring or not.

Statistics:
1. Nodes: 437929
2. Edges: 11373137

#### Citation
- Original Source
  - [Website](https://moleculenet.org)
  - LICENSE: MIT
```
@inproceedings{Wu2018Stanford,
  title={Moleculenet: a benchmark for molecular machine learning},
  author={Zhenqin Wu, Bharath Ramsundar, Evan N Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh SPappu, Karl Leswing, and Vijay Pande},
  booktitle={Chemical Science},
  year={2018},
  pages={513-530}
}
```

## Available Tasks
### [OGB](https://ogb.stanford.edu/)
- Task type: `GraphClassification`

#### Citation
```
@inproceedings{Hu2020Stanford,
  title={Strategies for pre-training graph neural networks. In International Conference on Learning Representations},
  year={2020},
  author={Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, and Jure Leskovec},
  booktitle={ICLR}
}
```

## Preprocessing
The data files and task config file in GLI format are transformed from the OGB implementation.

### Requirements
The preprocessing code requires the following package.
```
ogb >= 1.3.2
torch_geometric
numpy
torch
scipy
```
