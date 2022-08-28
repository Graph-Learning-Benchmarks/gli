# snap-patents

## Dataset Description
Snap-patents is a dataset of utility patents in the US. Each node is a patent, and edges connect patents that cite each other. Node features are derived from patent metadata. The task is set to predict the time at which a patent was granted, resulting in five classes.

Statistics:
- Nodes: 2923922
- Edges: 13975791
- Number of Classes: 5

#### Citation
- Original Source
  ```
  @misc{snapnets,
    author = {Jure Leskovec and Andrej Krevl},
    title= {{SNAP Datasets}: {Stanford} Large Network Dataset Collection},
    howpublished = {\url{http://snap.stanford.edu/data}},
    month = jun,
    year = 2014
  }
  ```

- Current Version
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

- Previous Version
  ```
  @inproceedings{leskovec2005graphs,
    title={Graphs over time: densification laws, shrinking diameters and possible explanations},
    author={Leskovec, Jure and Kleinberg, Jon and Faloutsos, Christos},
    booktitle={Proceedings of the eleventh ACM SIGKDD international conference on Knowledge discovery in data mining},
    pages={177--187},
    year={2005}
  }
  ```
## Available Tasks

### snap-patents

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
