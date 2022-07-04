# arXiv-year

## Dataset Description
arXiv-year is the ogbn-arXiv network with different labels. Our contribution is to set the class
labels to be the year that the paper is posted, instead of paper subject area. The nodes are arXiv
papers, and directed edges connect a paper to other papers that it cites. The node features are averaged
word2vec token features of both the title and abstract of the paper. The five classes are chosen by
partitioning the posting dates so that class ratios are approximately balanced

Statistics:
- Nodes: 169343
- Edges: 1166243
- Number of Classes: 5

#### Citation

```
@article{lim2021large,
  title={Large scale learning on non-homophilous graphs: New benchmarks and strong simple methods},
  author={Lim, Derek and Hohne, Felix and Li, Xiuyu and Huang, Sijia Linda and Gupta, Vaishnavi and Bhalerao, Omkar and Lim, Ser Nam},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={20887--20902},
  year={2021}
}
```
## Available Tasks

### arXiv-year

- Task type: `NodeClassification`


#### Citation

```
@article{lim2021large,
  title={Large scale learning on non-homophilous graphs: New benchmarks and strong simple methods},
  author={Lim, Derek and Hohne, Felix and Li, Xiuyu and Huang, Sijia Linda and Gupta, Vaishnavi and Bhalerao, Omkar and Lim, Ser Nam},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={20887--20902},
  year={2021}
}
```

## Preprocessing
The data file in GLB format is transformed from the [CUAI](https://github.com/CUAI/Non-Homophily-Large-Scale). Check `arXiv-year.ipynb` for the preprocessing.


### Requirements

The preprocessing code requires the following packages.

```
numpy==1.22.3
scipy==1.7.3
torch==1.11.0
torch_geometric==2.0.4
```
