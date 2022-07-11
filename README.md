# GLB-Repo

[![Pycodestyle](https://github.com/Graph-Learning-Benchmarks/GLB-Repo/actions/workflows/pycodestyle.yml/badge.svg)](https://github.com/Graph-Learning-Benchmarks/GLB-Repo/actions/workflows/pycodestyle.yml)
[![Pydocstyle](https://github.com/Graph-Learning-Benchmarks/GLB-Repo/actions/workflows/pydocstyle.yml/badge.svg)](https://github.com/Graph-Learning-Benchmarks/GLB-Repo/actions/workflows/pydocstyle.yml)
[![Pylint](https://github.com/Graph-Learning-Benchmarks/GLB-Repo/actions/workflows/pylint.yml/badge.svg)](https://github.com/Graph-Learning-Benchmarks/GLB-Repo/actions/workflows/pylint.yml)
[![Pytest](https://github.com/Graph-Learning-Benchmarks/GLB-Repo/actions/workflows/pytest.yml/badge.svg)](https://github.com/Graph-Learning-Benchmarks/GLB-Repo/actions/workflows/pytest.yml)

## Installation

```bash
pip install -e .
```

## Example of Data Loading

```bash
python3 example.py --task {NodeClassification,TimeDepenedentLinkPrediction,GraphClassification}
```

## Example of `metadata.json`

### Homogeneous Graph

A homogeneous graph contains the same type of nodes and the same type of edges.
Therefore, it can be considered as a special case of heterogeneous graph.
Compared with the data format of a heterogeneous graph, the structure
of `metadata.json` for homogeneous graphs have one fewer depth.

```json
{
    "description": "CORA dataset.",
    "data": {
        "Node": {
            "NodeFeature": {
                "description": "Node features of Cora dataset, 1/0-valued vectors.",
                "type": "int",
                "format": "SparseTensor",
                "file": "cora.npz",
                "key": "node_feats"
            },
            "NodeLabel": {
                "description": "Node labels of Cora dataset, int ranged from 1 to 7.",
                "type": "int",
                "format": "Tensor",
                "file": "cora.npz",
                "key": "node_class"
            }
        },
        "Edge": {
            "_Edge": {
                "file": "cora.npz",
                "key": "edge"
            }
        },
        "Graph": {
            "_NodeList": {
                "file": "cora.npz",
                "key": "node_list"
            },
            "_EdgeList": {
                "file": "cora.npz",
                "key": "edge_list"
            }
        }
    },
    "citation": "@inproceedings{yang2016revisiting,\ntitle={Revisiting semi-supervised learning with graph embeddings},\nauthor={Yang, Zhilin and Cohen, William and Salakhudinov, Ruslan},\nbooktitle={International conference on machine learning},\npages={40--48},\nyear={2016},\norganization={PMLR}\n}",
    "is_heterogeneous": false
}
```

### Heterogeneous Graph

A heterogeneous graph contains multiple kinds of nodes and edges. And we store
the attributes of different kinds of nodes and edges separately. Since the
features of different nodes and edges are stored separately, we need to assign
indices for each node and edge. Therefore, we require a `_ID` field to store
the indexing array in heterogeneous nodes and/or edges.

```json
{
    "description": "OGBN-MAG dataset.",
    "data": {
        "Node": {
            "PaperNode": {
                "_ID": {
                    "file": "ogbn-mag.npz",
                    "key": "PaperNode_id"
                },
                "PaperFeature": {
                    "description": "Node features of ogbn-mag dataset.",
                    "type": "float",
                    "format": "Tensor",
                    "file": "ogbn-mag.npz",
                    "key": "paper_feats"
                },
                "PaperLabel": {
                    "description": "Node labels of ogbn-mag dataset, int ranged from 1 to 40.",
                    "type": "int",
                    "format": "Tensor",
                    "file": "ogbn-mag.npz",
                    "key": "paper_class"
                },
                "PaperYear": {
                    "description": "Year of the article represented by the Node",
                    "type": "int",
                    "format": "Tensor",
                    "file": "ogbn-mag.npz",
                    "key": "paper_year"
                }
            },
            "AuthorNode": {
                "_ID": {
                    "file": "ogbn-mag.npz",
                    "key": "AuthorNode_id"
                }
            },
            "InstitutionNode": {
                "_ID": {
                    "file": "ogbn-mag.npz",
                    "key": "InstitutionNode_id"
                }
            },
            "FieldOfStudyNode": {
                "_ID": {
                    "file": "ogbn-mag.npz",
                    "key": "FieldOfStudyNode_id"
                }
            }
        },
        "Edge": {
            "Author_affiliated_with_Institution": {
                "_ID": {
                    "file": "ogbn-mag.npz",
                    "key": "author_institution_id"
                },
                "_Edge": {
                    "file": "ogbn-mag.npz",
                    "key": "author_institution_edge"
                }
            },
            "Author_writes_Paper": {
                "_ID": {
                    "file": "ogbn-mag.npz",
                    "key": "author_paper_id"
                },
                "_Edge": {
                    "file": "ogbn-mag.npz",
                    "key": "author_paper_edge"
                }
            },
            "Paper_cites_Paper": {
                "_ID": {
                    "file": "ogbn-mag.npz",
                    "key": "paper_paper_id"
                },
                "_Edge": {
                    "file": "ogbn-mag.npz",
                    "key": "paper_paper_edge"
                }
            },
            "Paper_has_topic_FieldOfStudy": {
                "_ID": {
                    "file": "ogbn-mag.npz",
                    "key": "paper_FieldOfStudy_id"
                },
                "_Edge": {
                    "file": "ogbn-mag.npz",
                    "key": "paper_FieldOfStudy_edge"
                }
            }
        },
        "Graph": {
            "_NodeList": {
                "file": "ogbn-mag.npz",
                "key": "node_list"
            },
            "_EdgeList": {
                "file": "ogbn-mag.npz",
                "key": "edge_list"
            }
        }
    },
    "citation": "@inproceedings{wang2020microsoft,\ntitle={Microsoft academic graph: When experts are not enough},\nauthor={Wang, Kuansan and Shen, Zhihong and Huang, Chiyuan and Wu, Chieh-Han and Dong, Yuxiao and Kanakia, Anshul},\nbooktitle={Quantitative Science Studies},\npages={396--413},\nyear={2020}\n}",
    "is_heterogeneous": true
}
```

## Example of `task.json`

There can be different tasks on the same dataset. Here is an example for node
classification (`NodeClassification`).

```json
{
    "description": "Node classification on CORA dataset. Planetoid split.",
    "type": "NodeClassification",
    "feature": [
        "Node/NodeFeature"
    ],
    "target": "Node/NodeLabel",
    "num_classes": 7,
    "train_set": {
        "file": "cora_task.npz",
        "key": "train"
    },
    "val_set": {
        "file": "cora_task.npz",
        "key": "val"
    },
    "test_set": {
        "file": "cora_task.npz",
        "key": "test"
    }
}
```

The features and targets for heterogeneous graphs are specified in a little
different way. For example, `Node/NodeSet1/NodeFeature`.

### Supported Task Types
1. `NodeClassification`
2. `TimeDependentLinkPrediction`
3. `GraphClassification`
