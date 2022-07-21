# Ogbl-ddi
## Data Description

The **ogbl-ddi** dataset is a homogeneous, unweighted, undirected graph, representing the drug-drug interaction network. Each node represents an FDA-approved or experimental drug. Edges represent interactions between drugs and can be interpreted as a phenomenon where the joint effect of taking the two drugs together is considerably different from the expected effect in which drugs act independently of each other.

Statistics:
1. Nodes: 4267
2. Edges: 2135822

#### Citation
- Original Source
```
@inproceedings{
  title={DrugBank 5.0: a major update to theDrugBank database for 2018},
  author={David S Wishart, Yannick D Feunang, An C Guo, Elvis J Lo, Ana Marcu, Jason R Grant, TanvirSajed, Daniel Johnson, Carin Li, Zinat Sayeeda, et al.},
  booktitle={Nucleic Acids Research},
  year={2018}
}
```
- Current Version
``` 
@inproceedings{
  author={Emre Guney},
  title={Reproducible drug repurposing: When similarity does not suffice},
  year={2017},
  pages={132=143}
  booktitle={Pacific Symposiumon Biocomputing}
}
```

## Available Tasks
### [OGB](https://ogb.stanford.edu)
- Task type: RelationLinkPrediction

#### Citation
``` 
@inproceedings{
  author={Emre Guney},
  title={Reproducible drug repurposing: When similarity does not suffice},
  year={2017},
  pages={132=143}
  booktitle={Pacific Symposiumon Biocomputing}
}
```

## Preprocessing
The data files and task config file in GLB format are transformed from the OGB implementation. 

### Requirements
The preprocessing code requires the following package.
```
ogb >= 1.2.1
```