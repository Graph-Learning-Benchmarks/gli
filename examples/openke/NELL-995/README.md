# NELL-995

## Dataset Description

The NELL-995 dataset contains knowledge base relation triples and textual mentions of Freebase entity pairs. It has a total of 592,213 triplets with 14,951 entities and 1,345 unique relationships.

Statistics:
- Nodes: 75492
- Edges: 154213

#### Citation

```
@article{xiong2017deeppath,
  title={Deeppath: A reinforcement learning method for knowledge graph reasoning},
  author={Xiong, Wenhan and Hoang, Thien and Wang, William Yang},
  journal={arXiv preprint arXiv:1707.06690},
  year={2017}
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
