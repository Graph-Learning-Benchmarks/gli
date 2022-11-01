# twitch-gamers

## Dataset Description
Twitch-gamers is a connected undirected graph of relationships between accounts on the streaming platform Twitch. Each node is a Twitch account, and edges exist between accounts that are mutual followers. The node features include number of views, creation and update dates, language, life time, and whether the account is dead. The binary classification task is to predict whether the channel has explicit content.

Statistics:
- Nodes: 168114
- Edges: 6797557
- Number of Classes: 2

#### Citation
- Original Source
  
  - [Website](https://github.com/benedekrozemberczki/datasets)
  - LICENSE: [MIT](https://github.com/benedekrozemberczki/datasets/blob/master/LICENSE)
  ```
  @article{rozemberczki2021twitch,
    title={Twitch gamers: a dataset for evaluating proximity preserving and structural role-based node embeddings},
    author={Rozemberczki, Benedek and Sarkar, Rik},
    journal={arXiv preprint arXiv:2101.03091},
    year={2021}
  }
  ```
- Current Version

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

### pokec

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
