# GLB-Repo

[![Pycodestyle](https://github.com/Graph-Learning-Benchmarks/GLB-Repo/actions/workflows/pycodestyle.yml/badge.svg)](https://github.com/Graph-Learning-Benchmarks/GLB-Repo/actions/workflows/pycodestyle.yml)
[![Pydocstyle](https://github.com/Graph-Learning-Benchmarks/GLB-Repo/actions/workflows/pydocstyle.yml/badge.svg)](https://github.com/Graph-Learning-Benchmarks/GLB-Repo/actions/workflows/pydocstyle.yml)
[![Pylint](https://github.com/Graph-Learning-Benchmarks/GLB-Repo/actions/workflows/pylint.yml/badge.svg)](https://github.com/Graph-Learning-Benchmarks/GLB-Repo/actions/workflows/pylint.yml)

## Example of `metadata.json`

### Homogeneous Graph

A homogeneous graph contains the same type of nodes and the same type of edges.
Therefore, it can be considered as a special case of heterogeneous graph.
Compared with the data format of a heterogeneous graph, the structure
of `metadata.json` for homogeneous graphs have one fewer depth.

```json
{
    "description": "An example of heterogeneous dataset.",
    "citation": "",
    "data": {
        "Node": {
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
        "Edge": {
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
3. `GraphPrediction`