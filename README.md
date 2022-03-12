# GLB-Repo
[![Pycodestyle](https://github.com/Graph-Learning-Benchmarks/GLB-Repo/actions/workflows/pycodestyle.yml/badge.svg)](https://github.com/Graph-Learning-Benchmarks/GLB-Repo/actions/workflows/pycodestyle.yml)
[![Pydocstyle](https://github.com/Graph-Learning-Benchmarks/GLB-Repo/actions/workflows/pydocstyle.yml/badge.svg)](https://github.com/Graph-Learning-Benchmarks/GLB-Repo/actions/workflows/pydocstyle.yml)
[![Pylint](https://github.com/Graph-Learning-Benchmarks/GLB-Repo/actions/workflows/pylint.yml/badge.svg)](https://github.com/Graph-Learning-Benchmarks/GLB-Repo/actions/workflows/pylint.yml)

## Example of `metadata.json`
```json
{
    "description": "An example of heterogeneous dataset.",
    "citation": "...",
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
                "NodeSet2Feature1": {...},
                "NodeSet2Feature2": {...},
                "NodeSet2Feature3": {...}
            },
            "NodeSet3": {
                "_ID": {
                    "file": "example.npz",
                    "key": "nodeset3_id"
                },
                "NodeSet3Feature1": {...}
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
                "_ID": {...},
                "_Edge": {...},
                "EdgeSet2Feature1": {...},
                "EdgeSet2Feature2": {...}
            },
            "EdgeSet3": {
                "_ID": {...},
                "_Edge": {...}
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