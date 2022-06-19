# WN11

## Dataset Description

The WN11 dataset contains knowledge base relation triples and textual mentions of Freebase entity pairs. It has a total of 592,213 triplets with 14,951 entities and 1,345 unique relationships.

Statistics:
- Nodes: 38588
- Edges: 125734

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

## Available Tasks
### Knowledge Graph Completion
+ Task type: `LinkPredictionEntity`
    - Predict the tail (head) entity given a pair of relation and head (tail).
+ Task type: `LinkPredictionRelation`
    - Predict the relation edge given a pair of head and tail entities.


### Requirements

The preprocessing code requires the following packages.

```
scipy==1.7.1
```
