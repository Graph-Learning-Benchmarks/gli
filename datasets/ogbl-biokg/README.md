# ogbl-biokg

## Dataset Description

The **ogbl-biokg** dataset is a Knowledge Graph (KG), which we created using data from a large number of biomedical data repositories. It contains 5 types of entities: diseases (10,687 nodes), proteins (17,499), drugs (10,533 nodes), side effects (9,969 nodes), and protein functions (45,085 nodes). There are 51 types of directed relations connecting two types of entities, including 38 kinds of drug-drug interactions, 8 kinds of protein-protein interaction, as well as drug-protein, drug-side effect, function-function relations. All relations are modeled as directed edges, among which the relations connecting the same entity types (e.g., protein-protein, drug-drug, function-function) are always symmetric, i.e., the edges are bi-directional.

Statistics:
- Nodes: 93773
- Edges: 4762678


## Available Tasks

### [OGB](https://ogb.stanford.edu/)

- Task type: `LinkPrediction`


## Preprocessing

The data files and task config file in GLB format are transformed from the OGB implementation. 


### Requirements

The preprocessing code requires the following packages.

```
ogb >= 1.2.0
```
