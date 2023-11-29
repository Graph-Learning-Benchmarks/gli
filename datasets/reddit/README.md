# Reddit

## Dataset Description

The Reddit dataset is a graph dataset from Reddit posts made in the month of September, 2014. The node label in this case is the community that a post belongs to. 50 large communities have been sampled to build a post-to-post graph, connecting posts if the same user comments on both. 

Statistics:
- Nodes: 232965
- Edges: 114615892
- Number of Classes: 41

#### Citation
- Original Source
  + [Website](http://snap.stanford.edu/graphsage/)
  + LICENSE: [MIT](https://github.com/williamleif/GraphSAGE/blob/master/LICENSE.txt)
```
@article{hamilton2017inductive,
  title={Inductive representation learning on large graphs},
  author={Hamilton, Will and Ying, Zhitao and Leskovec, Jure},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

- Current Version
  + [Website](http://snap.stanford.edu/graphsage/)
  + LICENSE: [MIT](https://github.com/williamleif/GraphSAGE/blob/master/LICENSE.txt)
```
@article{hamilton2017inductive,
  title={Inductive representation learning on large graphs},
  author={Hamilton, Will and Ying, Zhitao and Leskovec, Jure},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

- Previous Version




## Available Tasks

- Task type: `NodeClassification`


#### Citation

```
@article{hamilton2017inductive,
  title={Inductive representation learning on large graphs},
  author={Hamilton, Will and Ying, Zhitao and Leskovec, Jure},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

## Preprocessing

The data files and task config file in GLI format are transformed from the [DGL](https://www.dgl.ai) implementation (check docs for [Reddit Dataset](https://docs.dgl.ai/en/0.9.x/generated/dgl.data.RedditDataset.html?highlight=reddit#dgl.data.RedditDataset)). Check `reddit.ipynb` for the preprocessing.


### Requirements

The preprocessing code requires the following packages.

```
numpy
torch
dgl==1.1.2
```
