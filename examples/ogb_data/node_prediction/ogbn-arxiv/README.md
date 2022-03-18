# OGBN-ARXIV

## Dataset Description
The ogbn-arxiv dataset is a directed graph, representing the citation network between all Computer Science (CS) arXiv papers indexed by MAG [1]. Each node is an arXiv paper and each directed edge indicates that one paper cites another one. Each paper comes with a 128-dimensional feature vector obtained by averaging the embeddings of words in its title and abstract. The embeddings of individual words are computed by running the skip-gram model [2] over the MAG corpus. We also provide the mapping from MAG paper IDs into the raw texts of titles and abstracts here. In addition, all papers are also associated with the year that the corresponding paper was published.

[1] Kuansan Wang, Zhihong Shen, Chiyuan Huang, Chieh-Han Wu, Yuxiao Dong, and Anshul Kanakia. Microsoft academic graph: When experts are not enough. Quantitative Science Studies, 1(1):396–413, 2020.
[2] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representationsof words and phrases and their compositionality. In Advances in Neural Information Processing Systems (NeurIPS), pp. 3111–3119, 2013.


Statistics:
- Nodes: 169343
- Edges: 1166243
- Number of Classes: 40

#### Citation

```
@article{wang2020microsoft,
  title={Microsoft academic graph: When experts are not enough},
  author={Wang, Kuansan and Shen, Zhihong and Huang, Chiyuan and Wu, Chieh-Han and Dong, Yuxiao and Kanakia, Anshul},
  journal={Quantitative Science Studies},
  pages={396--413},
  year={2020}
}
```

## Available Tasks

### Planetoid

- Task type: `NodeClassification`

This is a node classification task with fixed split from [planetoid](https://github.com/kimiyoung/planetoid).

#### Citation

```
@inproceedings{wang2020microsoft,
    title={Microsoft academic graph: When experts are not enough},
    author={Wang, Kuansan and Shen, Zhihong and Huang, Chiyuan and Wu, Chieh-Han and Dong, Yuxiao and Kanakia, Anshul},
    booktitle={Quantitative Science Studies},
    pages={396--413},
    year={2020}
}
```

## Preprocessing

The data files and task config file in GLB format are transformed from the OGB implementation. 


### Requirements

The preprocessing code requires the following packages.

```
scipy==1.7.1
ogb==1.3.2
```
