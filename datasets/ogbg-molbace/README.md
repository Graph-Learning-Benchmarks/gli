# Ogbg-molbace

## Data Description

_Ogbg-molbace_ contains molecule graphs adopted from the MoleculeNet. The molecule graphs are converted from SMIE strings using RDKit. Each graph represents a molecule, where nodes are atoms, and edges are chemical bonds. Input node features are 9-dimensional, containing atomic number and chirality, as well as other additional atom features such as formal charge and whether the atom is in the ring or not.

Statistics:

- Nodes: 1513
- Edges: 223072

#### Citation

- Original Source
  - [Website](https://moleculenet.org)
  - LICENSE: [MIT](https://github.com/deepchem/deepchem/blob/master/LICENSE)

```
@inproceedings{Wu2018Stanford,
  title={Moleculenet: a benchmark for molecular machine learning},
  author={Zhenqin Wu, Bharath Ramsundar, Evan N Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh SPappu, Karl Leswing, and Vijay Pande},
  booktitle={Chemical Science},
  year={2018},
  pages={513-530}
}
```

- Current Version
  - [Website](https://github.com/snap-stanford/pretrain-gnns)
  - LICENSE: [MIT](https://github.com/snap-stanford/pretrain-gnns/blob/master/LICENSE)

#### Citation

```
@inproceedings{Hu2020Stanford,
  title={Strategies for pre-training graph neural networks. In International Conference on Learning Representations},
  author={Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, and Jure Leskovec},
  year={2020},
  booktitle={ICLR}
}
```

## Available Tasks

### [OGB](https://ogb.stanford.edu/)

- Task type: `GraphClassification`

#### Citation

```
@inproceedings{Hu2020Stanford,
  title={Strategies for pre-training graph neural networks. In International Conference on Learning Representations},
  author={Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, and Jure Leskovec},
  year={2020},
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
