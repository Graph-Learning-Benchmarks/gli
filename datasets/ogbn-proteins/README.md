# Ogbn-proteins
## Data Description

The **ogbn-proteins** dataset is an undirected, weighted, and typed (according to species) graph. Nodes represent proteins, and edges indicate different types of biologically meaningful associations between proteins, e.g., physical interactions, co-expression or homology. All edges come with 8-dimensional features, where each dimension represents the approximate confidence of a single association type and takes values between 0 and 1 (the larger the value is, the more confident we are about the association). The proteins come from 8 species.

Statistics:
1. Nodes: 132534
2. Edges: 79122504


#### Citation
- Original Source
  - [Website](https://github.com/related-sciences/string-protein-network)
  - LICENSE: [Apache](https://github.com/related-sciences/string-protein-network/blob/master/LICENSE.md)
```
@inproceedings{
  title={STRING v11: protein–protein association networks with increased coverage, supporting functional discovery in genome-wide experimental datasets.},
  author={Damian Szklarczyk, Annika L Gable, David Lyon, Alexander Junge, Stefan Wyder, Jaime Huerta- Cepas, Milan Simonovic, Nadezhda T Doncheva, John H Morris, Peer Bork, et al.},
  booktitle={Nucleic Acids Research},
  pages={607=613},
  year={2029}
}
```
- Previous Version
  - [Webstie](https://academic.oup.com/nar/article/47/D1/D330/5160994)
  - LICENSE: missing
```
@inproceedings{
  title={The gene ontology resource: 20 years and still going strong},
  author={TGene Ontology Consortium},
  booktitle={Nucleic Acids Research},
  pages={330=338},
  year={2018}
}
```

- Current Version
  - [Website](https://ogb.stanford.edu/docs/nodeprop/)
  - LICENSE: [Amazon](https://s3.amazonaws.com/amazon-reviews-pds/license.txt)
```
@article{hu2022stanford,
  title={Open Graph Benchmark: Datasets for Machine Learning on Graphs},
  author={Hu, Weihua and Fey, Matthias and Zitnik, Marinka and Dong, Yuxiao and Ren, Hongyu and Liu, Bowen and Catasta, Michele and Leskovec, Jure},
  year={2021}
}
```
## Available Tasks
### [OGB](https://ogb.stanford.edu/docs/nodeprop/)
- Task type: `NodeClassification`

#### Citation
```
@inproceedings{
  title={The gene ontology resource: 20 years and still going strong},
  author={TGene Ontology Consortium},
  booktitle={Nucleic Acids Research},
  pages={330=338},
  year={2018}
}
```

## Preprocessing
The data files and task config file in GLI format are transformed from the OGB implementation.

### Requirements
The preprocessing code requires the following package.
```
ogb >= 1.1.1
```
