# Ogbg-molfreesolv
## Data Description

**Ogbg-molfreesolv** is adopted from the MoleculeNet. It is a smaller dataset used for regression. All the molecules are pre-processed using RDKit. Each graph represents a molecule, where nodes are atoms, and edges are chemical bonds. Input node features are 9-dimensional, containing atomic number and chirality, as well as other additional atom features such as formal charge and whether the atom is in the ring or not. The full description of the features is provided in code. The script to convert the SMILES string to the above graph object can be found here. Note that the script requires RDkit to be installed. The script can be used to pre-process external molecule datasets so that those datasets share the same input feature space as the OGB molecule datasets. This is particularly useful for pre-training graph models, which has great potential to significantly increase generalization performance on the (downstream) OGB datasets.

Statistics:
1. Nodes: 5600
2. Edges: 642

### Citation
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
### Task type
`GraphClassification`

## Preprocessing
The data files and task config file in GLB format are transformed from the OGB implementation. 

### Requirements
The preprocessing code requires the following package.

> ogb>= 1.3.2

### Citation
``` 
@inproceedings{
  author={Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, and Jure Leskovec},
  title={Strategies for pre-training graph neural networks. In International Conference on Learning Representations},
  year={2020},
  booktitle={ICLR}
}