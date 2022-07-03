# OGBG-MOLHIV

## Dataset Description
The ogbg-molhiv contains molecule graphs adopted from the MoleculeNet. The molecule graphs are converted from SMIE strings using RDKit. Each graph represents a molecule, where nodes are atoms, and edges are chemical bonds. Input node features are 9-dimensional, containing atomic number and chirality, as well as other additional atom features such as formal charge and whether the atom is in the ring or not.


Statistics:
- Graph: 41,127
- Nodes per graph: 25.5
- Edges per graph: 27.5

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

Data file(s) is transformed from OGB, Check ogbg-molhiv.ipynb for preprocessing.

### Requirements

The preprocessing code requires the following packages.

```
scipy==1.7.1
ogb==1.3.2
```
