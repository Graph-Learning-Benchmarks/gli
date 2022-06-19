# WN18

## Dataset Description

The WN18 dataset contains knowledge base relation triples and textual mentions of Freebase entity pairs. It has a total of 592,213 triplets with 14,951 entities and 1,345 unique relationships.

Statistics:
- Nodes: 40943
- Edges: 151442

#### Citation

```
@article{bordes2013translating,
  title={Translating embeddings for modeling multi-relational data},
  author={Bordes, Antoine and Usunier, Nicolas and Garcia-Duran, Alberto and Weston, Jason and Yakhnenko, Oksana},
  journal={Advances in neural information processing systems},
  volume={26},
  year={2013}
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
