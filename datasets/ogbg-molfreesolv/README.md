# Ogbg-molfreesolv
## Data Description

*Ogbg-molfreesolv* contains molecule graphs adopted from the MoleculeNet. The molecule graphs are converted from SMIE strings using RDKit. Each graph represents a molecule, where nodes are atoms, and edges are chemical bonds. Input node features are 9-dimensional, containing atomic number and chirality, as well as other additional atom features such as formal charge and whether the atom is in the ring or not.

Statistics:
- Nodes: 5600
- Edges: 642

#### Citation
```
@inproceedings{
  author={Zhenqin Wu, Bharath Ramsundar, Evan N Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh SPappu, Karl Leswing, and Vijay Pande}, 
  title={Moleculenet: a benchmark for molecular machine learning},
  booktitle={Chemical Science},
  year={2018},
  pages={513–530}
}
```

## Available Tasks
### OGB

- Task type:  `GraphRegression`

#### Citation
``` 
@inproceedings{
  author={Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, and Jure Leskovec},
  title={Strategies for pre-training graph neural networks. In International Conference on Learning Representations},
  year={2020},
  booktitle={ICLR}
}
```

## Preprocessing
The data files and task config file in GLB format are transformed from the OGB implementation. 

### Requirements
The preprocessing code requires the following package.

```
ogb >= 1.3.2
```
