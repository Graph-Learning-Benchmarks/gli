# OGBL-COLLAB

## Dataset Description
The ogbg-molhiv and ogbg-molpcba datasets are two molecular property prediction datasets of different sizes: ogbg-molhiv (small) and ogbg-molpcba (medium). They are adopted from the MoleculeNet, and are among the largest of the MoleculeNet datasets. All the molecules are pre-processed using RDKit. Each graph represents a molecule, where nodes are atoms, and edges are chemical bonds. Input node features are 9-dimensional, containing atomic number and chirality, as well as other additional atom features such as formal charge and whether the atom is in the ring or not. The full description of the features is provided in code. The script to convert the SMILES string to the above graph object can be found here. Note that the script requires RDkit to be installed. The script can be used to pre-process external molecule datasets so that those datasets share the same input feature space as the OGB molecule datasets. This is particularly useful for pre-training graph models, which has great potential to significantly increase generalization performance on the (downstream) OGB datasets.



Statistics:
- Graph: 41,127
- Nodes per graph: 25.5
- Edges per graph: 27.5

#### Citation

```
@article{hu2022stanford,
  title={Open Graph Benchmark: Datasets for Machine Learning on Graphs},
  author={Hu, Weihua and Fey, Matthias and Zitnik, Marinka and Dong, Yuxiao and Ren, Hongyu and Liu, Bowen and Catasta, Michele and Leskovec, Jure},
  year={2021}
}
```

## Available Tasks

### Planetoid

- Task type: `GraphClassification`


#### Citation

```
@article{hu2022stanford,
  title={Open Graph Benchmark: Datasets for Machine Learning on Graphs},
  author={Hu, Weihua and Fey, Matthias and Zitnik, Marinka and Dong, Yuxiao and Ren, Hongyu and Liu, Bowen and Catasta, Michele and Leskovec, Jure},
  year={2021}
}
```

## Preprocessing

Check ogbg-molhiv.ipynb for preprocessing

### Requirements

The preprocessing code requires the following packages.

```
scipy==1.7.1
ogb==1.3.2
```
