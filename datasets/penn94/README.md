# Penn94
## Dataset Description

Penn94 is a friendship network from the Facebook 100 networks of university students from 2005, where nodes represent students. Each node is labeled with the reported gender of the user. The node features are major, second major/minor, dorm/house, year, and high school.

Statistics:
- Nodes: 41554
- Edges: 2724458
- Number of Classes: 2

#### Citation
- Original Source
  ```
  @article{TRAUD20124165,
    title = {Social structure of Facebook networks},
    author = {Amanda L. Traud and Peter J. Mucha and Mason A. Porter},
    journal = {Physica A: Statistical Mechanics and its Applications},
    volume = {391},
    number = {16},
    pages = {4165-4180},
    year = {2012},
    issn = {0378-4371},
    doi = {https://doi.org/10.1016/j.physa.2011.12.021},
    url = {https://www.sciencedirect.com/science/article/pii/S0378437111009186}
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
## Available Tasks

### Penn94

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
