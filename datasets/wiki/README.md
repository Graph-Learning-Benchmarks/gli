# wiki

## Dataset Description
Wiki is a dataset of Wikipedia articles, where nodes represent pages and edges represent links between them. This dataset is collected by Lim<sup>[1](#myfootnote1)</sup>. Node features are constructed using averaged title and abstract GloVe embeddings. Labels represent total page views over 60 days, which are partitioned into quintiles to make five classes.

Statistics:
- Nodes: 1925342
- Edges: 303434860
- Number of Classes: 5

<a name="myfootnote1">[1]</a>: Lim, Derek, Felix Hohne, Xiuyu Li, Sijia Linda Huang, Vaishnavi Gupta, Omkar Bhalerao, and Ser Nam Lim. "Large scale learning on non-homophilous graphs: New benchmarks and strong simple methods." Advances in Neural Information Processing Systems 34 (2021): 20887-20902.


#### Citation
- Original Source
  
  - [Website](https://github.com/CUAI/Non-Homophily-Large-Scale)
  - LICENSE: missing
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

### wiki

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
The data file in GLI format is transformed from the [CUAI](https://github.com/CUAI/Non-Homophily-Large-Scale). Check [Non-homo-datasets](https://github.com/GreatSnoopyMe/Non-homo-datasets) for the preprocessing.


### Requirements

The preprocessing code requires the following packages.

```
dataset==1.5.2
numpy==1.22.3
scipy==1.7.3
torch==1.11.0
```
