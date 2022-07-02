# Ogbg-molfreesolv
## Data Description

The ogbg-molhiv and ogbg-molpcba datasets are two molecular property prediction datasets of different sizes: ogbg-molhiv (small) and ogbg-molpcba (medium). They are adopted from the MoleculeNet [1], and are among the largest of the MoleculeNet datasets. All the molecules are pre-processed using RDKit. Each graph represents a molecule, where nodes are atoms, and edges are chemical bonds. Input node features are 9-dimensional, containing atomic number and chirality, as well as other additional atom features such as formal charge and whether the atom is in the ring or not. The full description of the features is provided in code. The script to convert the SMILES string to the above graph object can be found here. Note that the script requires RDkit to be installed. The script can be used to pre-process external molecule datasets so that those datasets share the same input feature space as the OGB molecule datasets. This is particularly useful for pre-training graph models, which has great potential to significantly increase generalization performance on the (downstream) OGB datasets.

Beside the two main datasets, we additionally provide 10 smaller datasets from MoleculeNet. They are ogbg-moltox21, ogbg-molbace, ogbg-molbbbp, ogbg-molclintox, ogbg-molmuv, ogbg-molsider, and ogbg-moltoxcast for (multi-task) binary classification, and ogbg-molesol, **ogbg-molfreesolv**, and ogbg-mollipo for regression. 

Statistics:
1. Nodes: 5600
2. Edges: 10770

### Citation
> @inproceedings{\nauthor={Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, and Jure Leskovec},\ntitle={Strategies for pre-training graph neural networks. In International Conference on Learning Representations},\nyear={2020},\nbooktitle={ICLR}}

## Available Tasks
### Planetoid
Task type:

## Preprocessing
The data files and task config file in GLB format are transformed from the OGB implementation.

### Requirements
The preprocessing code requires the following package.

> ogb==1.3.2