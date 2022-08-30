# genius

## Dataset Description
Genius is a subset of the social network on genius.com — a site for crowdsourced annotations of song lyrics. Nodes are users, and edges connect users that follow each other on the site. The node classification task on this dataset is to predict certain marks on the accounts. About 20% of users in the dataset are marked “gone” on the site, which appears to often include spam users. Thus, nodes are predicted to be whether marked or not. The node features are user usage attributes like the Genius assigned expertise score, counts of contributions, and roles held by the user.

Statistics:
- Nodes: 421961
- Edges: 984979
- Number of Classes: 2

#### Citation
- Original Source
  ```
  @inproceedings{lim2021expertise,
    title={Expertise and Dynamics within Crowdsourced Musical Knowledge Curation: A Case Study of the Genius Platform.},
    author={Lim, Derek and Benson, Austin R},
    booktitle={ICWSM},
    pages={373--384},
    year={2021}
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

### genius

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
The data file in GLB format is transformed from the [CUAI](https://github.com/CUAI/Non-Homophily-Large-Scale). Check [Non-homo-datasets](https://github.com/GreatSnoopyMe/Non-homo-datasets) for the preprocessing.


### Requirements

The preprocessing code requires the following packages.

```
dataset==1.5.2
numpy==1.22.3
scipy==1.7.3
torch==1.11.0
```
