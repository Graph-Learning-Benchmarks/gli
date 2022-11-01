# YAGO3-10

## Dataset Description

The YAGO3-10 dataset is a subset of YAGO3 that only contains entities with at least 10 relations. It is a large semantic knowledge base, derived from Wikipedia, WordNet, WikiData, GeoNames, and other data sources It has a total of 1,089,040 triplets with 123,182 entities and 37 unique relationships.

Statistics:
- Nodes: 123182
- Edges: 1089040

#### Citation
- Original Source
	- [Website](https://yago-knowledge.org/)
	- LICENSE: [MIT](https://creativecommons.org/licenses/by/4.0/)
```
@inproceedings{suchanek2007yago,
    title={Yago: a core of semantic knowledge},
    author={Suchanek, Fabian M and Kasneci, Gjergji and Weikum, Gerhard},
    booktitle={Proceedings of the 16th international conference on World Wide Web},
    pages={697--706},
    year={2007}
}
```
- Current Version
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
- Previous Version
	- [Website](https://yago-knowledge.org/downloads/yago-3)
	- LICENSE: missing
```
@inproceedings{mahdisoltani2014yago3,
    title={Yago3: A knowledge base from multilingual wikipedias},
    author={Mahdisoltani, Farzaneh and Biega, Joanna and Suchanek, Fabian},
    booktitle={7th biennial conference on innovative data systems research},
    year={2014},
    organization={CIDR Conference}
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
