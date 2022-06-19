# FB15K

## Dataset Description

The FB15K dataset contains knowledge base relation triples and textual mentions of Freebase entity pairs. It has a total of 592,213 triplets with 14,951 entities and 1,345 unique relationships.

Statistics:
- Nodes: 40943
- Edges: 93003

#### Citation

```
@inproceedings{dettmers2018convolutional,
  title={Convolutional 2d knowledge graph embeddings},
  author={Dettmers, Tim and Minervini, Pasquale and Stenetorp, Pontus and Riedel, Sebastian},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={32},
  number={1},
  year={2018}
}
```

## Available Tasks
### Knowledge Graph Completion
+ Task type: `LinkPredictionEntity`
    - Predict the tail (head) entity given a pair of relation and head (tail).
+ Task type: `LinkPredictionRelation`
    - Predict the relation edge given a pair of head and tail entities.


### Requirements

The preprocessing code requires the following packages.

```
scipy==1.7.1
```
