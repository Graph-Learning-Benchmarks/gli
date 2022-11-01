# NELL-995

## Dataset Description

The NELL-995 dataset is a subset of the NELL dataset which captures two types of knowledge: (1) knowledge about which noun phrases refer to which specified semantic categories, such as cities, companies, and sports teams, and (2) knowledge about which pairs of noun phrases satisfy which specified semantic relations, such as hasOfficesIn(organization, location). It has a total of 154,213 triplets with 75,492 entities and 200 unique relationships.

Statistics:
- Nodes: 75492
- Edges: 154213

#### Citation
- Original Source
	- [Website](https://aclanthology.org/D17-1060/)
	- LICENSE: missing
```
@article{xiong2017deeppath,
  title={Deeppath: A reinforcement learning method for knowledge graph reasoning},
  author={Xiong, Wenhan and Hoang, Thien and Wang, William Yang},
  journal={arXiv preprint arXiv:1707.06690},
  year={2017}
}
```
- Current Version
	- [Website](https://dl.acm.org/doi/abs/10.1016/j.websem.2019.01.004)
	- LICENSE: missing
```
@article{padia2019knowledge,
    title={Knowledge graph fact prediction via knowledge-enriched tensor factorization},
    author={Padia, Ankur and Kalpakis, Konstantinos and Ferraro, Francis and Finin, Tim},
    journal={Journal of Web Semantics},
    volume={59},
    pages={100497},
    year={2019},
    publisher={Elsevier}
}
```
- Previous Version
	- [Website](https://github.com/thunlp/OpenKE)
	- LICENSE: missing
```
@inproceedings{han2018openke,
    title={OpenKE: An Open Toolkit for Knowledge Embedding},
    author={Han, Xu and Cao, Shulin and Lv Xin and Lin, Yankai and Liu, Zhiyuan and Sun, Maosong  and Li, Juanzi},
    booktitle={Proceedings of EMNLP},
    year={2018}
}
```
## Available Tasks

### Knowledge Graph Completion

+ Task type: `KGEntityPrediction`
    - Predict the tail (head) entity given a pair of relation and head (tail).
+ Task type: `KGRelationPrediction`
    - Predict the relation edge given a pair of head and tail entities.

#### Citation

##### Link Prediction Task

```
@article{padia2019knowledge,
    title={Knowledge graph fact prediction via knowledge-enriched tensor factorization},
    author={Padia, Ankur and Kalpakis, Konstantinos and Ferraro, Francis and Finin, Tim},
    journal={Journal of Web Semantics},
    volume={59},
    pages={100497},
    year={2019},
    publisher={Elsevier}
}
```

##### Train, Validation, Test Split

```
@inproceedings{han2018openke,
    title={OpenKE: An Open Toolkit for Knowledge Embedding},
    author={Han, Xu and Cao, Shulin and Lv Xin and Lin, Yankai and Liu, Zhiyuan and Sun, Maosong  and Li, Juanzi},
    booktitle={Proceedings of EMNLP},
    year={2018}
}
```

## Preprocessing

The data files and task config file in GLI format are transformed from the [OpenKE](https://github.com/thunlp/OpenKE) implementation.

### Requirements

The preprocessing code requires the following packages.

```
scipy==1.7.1
```
