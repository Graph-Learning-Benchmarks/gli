# GLB-Repo

[![Pycodestyle](https://github.com/Graph-Learning-Benchmarks/GLB-Repo/actions/workflows/pycodestyle.yml/badge.svg)](https://github.com/Graph-Learning-Benchmarks/GLB-Repo/actions/workflows/pycodestyle.yml)
[![Pydocstyle](https://github.com/Graph-Learning-Benchmarks/GLB-Repo/actions/workflows/pydocstyle.yml/badge.svg)](https://github.com/Graph-Learning-Benchmarks/GLB-Repo/actions/workflows/pydocstyle.yml)
[![Pylint](https://github.com/Graph-Learning-Benchmarks/GLB-Repo/actions/workflows/pylint.yml/badge.svg)](https://github.com/Graph-Learning-Benchmarks/GLB-Repo/actions/workflows/pylint.yml)

## Installation

```bash
python setup.py install
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
    "citation": "@inproceedings{wang2020microsoft,\ntitle={Microsoft academic graph: When experts are not enough},\nauthor={Wang, Kuansan and Shen, Zhihong and Huang, Chiyuan and Wu, Chieh-Han and Dong, Yuxiao and Kanakia, Anshul},\nbooktitle={Quantitative Science Studies},\npages={396--413},\nyear={2020}\n}"
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
    "description": "An example of heterogeneous dataset.",
    "citation": "",
    "data": {
        "Node": {
            "NodeSet1": {
                "_ID": {
                    "file": "example.npz",
                    "key": "nodeset1_id"
                },
                "NodeSet1Feature1": {
                    "description": "Float node features of NodeSet1.",
                    "type": "float",
                    "format": "Tensor",
                    "file": "example.npz",
                    "key": "nodeset1_feat1"
                },
                "NodeSet1Feature2": {
                    "description": "Int node features of NodeSet1.",
                    "type": "int",
                    "format": "SparseTensor",
                    "file": "example.npz",
                    "key": "nodeset1_feat2"
                }
            },
            "NodeSet2": {
                "_ID": {
                    "file": "example.npz",
                    "key": "nodeset2_id"
                },
                "NodeSet2Feature1": {},
                "NodeSet2Feature2": {},
                "NodeSet2Feature3": {}
            },
            "NodeSet3": {
                "_ID": {
                    "file": "example.npz",
                    "key": "nodeset3_id"
                },
                "NodeSet3Feature1": {}
            }
        },
        "Edge": {
            "EdgeSet1": {
                "_ID": {
                    "file": "example.npz",
                    "key": "edgeset1_id"
                },
                "_Edge": {
                    "file": "example.npz",
                    "key": "edge1"
                },
                "EdgeSet1Feature1": {
                    "description": "Edge creation year.",
                    "type": "int",
                    "format": "Tensor",
                    "file": "example.npz",
                    "key": "edgeset1_feat1"
                }
            },
            "EdgeSet2": {
                "_ID": {},
                "_Edge": {},
                "EdgeSet2Feature1": {},
                "EdgeSet2Feature2": {}
            },
            "EdgeSet3": {
                "_ID": {},
                "_Edge": {}
            }
        },
        "Graph": {
            "_NodeList": {
                "file": "example.npz",
                "key": "node_list"
            },
            "_EdgeList": {
                "file": "example.npz",
                "key": "edge_list"
            }
        }
    }
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
