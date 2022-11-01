# FB15K237

## Dataset Description

The FB15K237 dataset contains knowledge base relation triples and textual mentions of Freebase entity pairs. It has a total of 310,116 triplets with 14,541 entities and 237 unique relationships.

Statistics:
- Nodes: 14541
- Edges: 310116

#### Citation

```
@inproceedings{10.1145/1376616.1376746,
    author = {Bollacker, Kurt and Evans, Colin and Paritosh, Praveen and Sturge, Tim and Taylor, Jamie},
    title = {Freebase: A Collaboratively Created Graph Database for Structuring Human Knowledge},
    year = {2008},
    isbn = {9781605581026},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/1376616.1376746},
    doi = {10.1145/1376616.1376746},
    abstract = {Freebase is a practical, scalable tuple database used to structure general human knowledge. The data in Freebase is collaboratively created, structured, and maintained. Freebase currently contains more than 125,000,000 tuples, more than 4000 types, and more than 7000 properties. Public read/write access to Freebase is allowed through an HTTP-based graph-query API using the Metaweb Query Language (MQL) as a data query and manipulation language. MQL provides an easy-to-use object-oriented interface to the tuple data in Freebase and is designed to facilitate the creation of collaborative, Web-based data-oriented applications.},
    booktitle = {Proceedings of the 2008 ACM SIGMOD International Conference on Management of Data},
    pages = {1247–1250},
    numpages = {4},
    keywords = {collaborative systems, semantic network, tuple store},
    location = {Vancouver, Canada},
    series = {SIGMOD '08}
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
	- [Website](https://aclanthology.org/D15-1174/)
	- LICENSE: missing
```
@inproceedings{toutanova2015representing,
    title={Representing text for joint embedding of text and knowledge bases},
    author={Toutanova, Kristina and Chen, Danqi and Pantel, Patrick and Poon, Hoifung and   Choudhury, Pallavi and Gamon, Michael},
    booktitle={Proceedings of the 2015 conference on empirical methods in natural language  processing},
    pages={1499--1509},
    year={2015}
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
