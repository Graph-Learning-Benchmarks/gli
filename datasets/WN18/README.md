# WN18

## Dataset Description

WordNet is a large lexical database of English. Nouns, verbs, adjectives and adverbs are grouped into sets of cognitive synonyms (synsets), each expressing a distinct concept. The WN18 dataset is a subset of WordNet. It has a total of 151,442 triplets with 40,943 entities and 18 unique relationships.

Statistics:
- Nodes: 40943
- Edges: 151442

#### Citation
- Original Source
```
@book{miller1998wordnet,
    title={WordNet: An electronic lexical database},
    author={Miller, George A},
    year={1998},
    publisher={MIT press}
}
```
- Current Version
```
@inproceedings{han2018openke,
    title={OpenKE: An Open Toolkit for Knowledge Embedding},
    author={Han, Xu and Cao, Shulin and Lv Xin and Lin, Yankai and Liu, Zhiyuan and Sun, Maosong  and Li, Juanzi},
    booktitle={Proceedings of EMNLP},
    year={2018}
}
```
- Previous Version
```
@article{bordes2013translating,
  title={Translating embeddings for modeling multi-relational data},
  author={Bordes, Antoine and Usunier, Nicolas and Garcia-Duran, Alberto and Weston, Jason and Yakhnenko, Oksana},
  journal={Advances in neural information processing systems},
  volume={26},
  year={2013}
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
