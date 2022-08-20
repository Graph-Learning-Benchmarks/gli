# GLI Data and Task Format

The central component of the data standardizing pipeline is a definition of the standard GLI file format. The dataset contributors are supposed to convert the raw data into this standard format and submit their dataset in the GLI format. This platform will provide some helper functions to ease this process in the future. This platform will also provide tools to construct Dataset class in popular graph learning libraries out of the GLI format data files.

![flowchart](/img/flowchart.png)

## Design Objectives

There are several principles for the design of the GLI file format.

1. Allow general dataloader that can be widely applicable to different datasets on different tasks.
2. Build consistent APIs with downstream graph learning libraries such as [`dgl`](https://www.dgl.ai).
3. Design generic format that is flexible to accommodate various graph-structured data.
4. Allow efficient storage and fast access.

### Explicit separation of data storage and task configuration

GLI makes an explicit separation between the data storage and the task configuration for graph learning. i.e., Multiple tasks can be performed on the same dataset, or the same task can be performed on different datasets. The separation between graphs and tasks further allows users to use general datasets bound to every type of task that can be applied to every graph dataset.

### Objects with attribute scheme

GLI aims to accommodate various types of graph-structured data, such as multi-graphs, heterogeneous graphs, dynamic graphs, or even hypergraphs, as well as graphs with node attributes and edge attributes.

For this purpose, we treat nodes, edges, and graphs all as objects with attributes. In particular, edges are objects with a special attribute indicating the two end nodes; graphs are objects with a special attribute indicating the list of nodes it contains. Treating the edges and graphs as objects with special attributes makes it easy to extend to different types of data.

However, being too flexible about what attributes can be included will make it difficult for general dataloading. We therefore require all the attributes to follow a set of predefined attribute types. Currently, these data types are allowed for attributes:
1. `float`
2. `int`
3. `string`

When the existing attribute types are not enough to accommodate a certain type of data, we can further expand the predefined attribute types.

### Efficient storage

Tensor is a common storage scheme for graph data. GLI also provides separate supporting data files for efficient storage of certain attributes such as high-dim sparse features. For example, the `NodeFeature` attribute of `cora` dataset is stored in the format of `SparseTensor`. Currently, these data formats are allowed for attributes:
1. `Tensor`
2. `SparseTensor` (In particular, only `csr` and `coo` tensor are stably supported.)

When the existing attribute formats are not enough to accommodate a certain type of data, we can further expand the predefined attribute types.

## GLI Data Format

The metadata of the graph dataset should be stored in a file named as `metadata.json`, which contains the pointers to various npz files storing the actual data. There are four required fields in `metadata.json`.

- `description`: A short description of the dataset, typically the dataset name.
- `data`: The graph data as well as their attributes.
- `citation`: The bibtex of the original dataset.
- `is_heterogeneous`: A boolean value that indicates the graph heterogeneity.

Since GLI is designed to separate tasks from graph data, this file should not contain any information regarding task configuration, e.g., negative edges in link prediction tasks.

### Overview

GLI predefine 3 **objects**: `Node`, `Edge`, and `Graph` in its framework. Each objects may have multiple **attributes**, e.g., features and prediction targets. There are three reserved attributes, though: `_Edge`, `_NodeList`, and `_EdgeList`. Except for them, users are free to define any extra properties.

#### Reserved attributes

As mentioned above, there are three reserved attributes in GLI, all of which starts with `_`: `_Edge`, `_NodeList`, and `_EdgeList`. They store essential structural information about the graph data and should be placed under different objects as shown below:

- Node
  - No required attribute.
- Edge
  - `_Edge` (*required*): A `Tensor` in shape `(n_edges, 2)`. `_Edge[i]` stores the i-th edge in the form of `(src_node_id, dst_node_id)`.
- Graph
  - `_NodeList` (*required*): A 0/1-valued `Tensor`/`SparseTensor` in shape `(n_graphs, n_nodes)`. 
  - `_EdgeList` (*optional*): A 0/1-valued `Tensor`/`SparseTensor` in shape `(n_graphs, n_edges)`. This attribute is optional when the `_EdgeList` of each subgraph can be inferred from `_NodeList`.

#### Attribute properties

For reserved attributes, contributors only need to provide the data location file (may with key in a npz file). For extra user-defined attributes, contributors need to provide some extra information for loading (`description`, `type`, `format`).

- `description`: A description of the user-defined attribute.
- `type`: The data type of the attribute. GLI currently supports `int`, `float`, and `string`. We plan to include more types like media in the future.
- `format`: The data format of the attribute. GLI currently supports `Tensor` and `SparseTensor`. We plan to include more formats like mp3 in the future.
- `file`: The relative path of a file that contains the attribute. Currently, GLI supports `npy` and `npz` files.
- `key`: The key that indexes the attribute in dictionary-like file format like `npz` files.

#### Example (homogeneous graph)

A complete example of a homogeneous graph's `metadata.json` is given below.

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

#### Heterogeneous graph

Heterogeneous graph has multiple kinds of nodes and edges. For a heterograph, `Node` and `Edge` are dictionaries that map node/edge group name to its attributes. In addition to the required attributes that homogeneous graph needs, metadata of a heterogeneous graph requires these attributes:

- Node
	- `_ID`: The unique indices for all nodes.
- Edge
	- `_ID`: The unique indices for all edges.

Both `_ID`s should be indexed from 0.

#### Example (heterogeneous graph)

A complete example of a heterogeneous graph's `metadata.json` is given below.

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

## GLB Task Format

The information about a graph learning task (e.g., the train/test splits or the prediction target) should be stored in a *task configuration file* named as `task_<task_type>.json`. There could be multiple different tasks for a single graph dataset, such as node classification and link prediction. Node classification with different data split can also be viewed as different tasks.

### Supported Tasks

GLI currently supports following task types:

#### `NodeClassification`

- Description: This task requires the model to perform classification on each node.
- The list of required keys in the task configuration file:
    - `description`: task description.
    - `type`: task type (in this case, `NodeClassification`).
    - `feature`: the node attribute used as node feature in this task.
    - `target`: the node attribute used as prediction target in this task.
    - `num_classes`: the number of classes.
    - `train_set` (optional): the training node IDs.
    - `val_set` (optional): the validation node IDs.
    - `test_set` (optional): the test node IDs.
    - `num_splits` (optional): the number of data splits.
    - `train_ratio` (optional): the ratio of train set size in `num_sampels`.
    - `val ratio` (optional): the ratio of validation set size in `num_sampels`.
    - `test_ratio` (optional): the ratio of test set size in `num_sampels`.
    - `num_samples` (optional): total number of samples.
- There could be three types of data split. 
    - Fixed split: A fixed data split is associated with the task. Task configuration file should specify `train_set`, `val_set` and `test_set`. `num_splits` should be set to 1 or omitted.
    - Fixed multi-split: Multiple fixed data splits are associated with the task. Task configuration file should specify `train_set`, `val_set`, `test_set` and `num_splits`.
    - Random split: No fixed data splits come with the task. Task configuration file should specify `train_ratio`, `val_ratio`, `test_ratio` and `num_samples`.

#### `GraphClassification`

- Description: This task requires the model to perform classification on each graph.
- The list of required keys in task configuration file:
    - `description`: task description.
    - `type`: task type (in this case, `GraphClassification`).
    - `feature`: the attribute(s) used as feature in this task.
    - `target`: the graph attribute used as prediction target in this task.
    - `train_set` (optional): the training node IDs.
    - `val_set` (optional): the validation node IDs.
    - `test_set` (optional): the test node IDs.
    - `num_splits` (optional): the number of data splits.
    - `train_ratio` (optional): the ratio of train set size in `num_sampels`.
    - `val ratio` (optional): the ratio of validation set size in `num_sampels`.
    - `test_ratio` (optional): the ratio of test set size in `num_sampels`.
    - `num_samples` (optional): total number of samples.
- There could be two types of data split (fixed multi-split not supported). 
    - Fixed split: A fixed data split is associated with the task. Task configuration file should specify `train_set`, `val_set` and `test_set`. `num_splits` should be set to 1 or omitted.
    - Random split: No fixed data splits come with the task. Task configuration file should specify `train_ratio`, `val_ratio`, `test_ratio` and `num_samples`.

#### `LinkPrediction`
- Description: This task requires the model to perform link prediction on a graph.
    - The list of required keys in task configuration file:
        - `description`: task description.
        - `type`: task type (in this case, `LinkPrediction`).
        - `feature`: the attribute(s) used as feature in this task.
        - `train_set` (optional): the training edge IDs.
        - `val_set` (optional): the validation edge IDs.
        - `test_set` (optional): the test edge IDs.
        - `valid_neg` (optional): the negative samples of edges to validate.
        - `test_neg` (optional): the negative samples of edges to test.

#### `TimeDependentLinkPrediction`
- Description: This task requires the model to perform link prediction on a graph. The dataset is split according to time.
- The list of required keys in task configuration file:
    - `description`: task description.
    - `type`: task type (in this case, `TimeDependentLinkPrediction`).
    - `feature`: the attribute(s) used as feature in this task.
    - `time`: the time attribute that indicates edge formation order in this task.
    - `train_time_window`: the time window in which edges are used to train.
    - `val_time_window`: the time window in which edges are used to validate.
    - `test_time_window`: the time window in which edges are used to test.
    - `valid_neg` (optional): the negative samples of edges to validate.
    - `test_neg` (optional): the negative samples of edges to test.

#### `KGEntityPrediction`
- Description: This task requires the model to predict the tail or head node for a triplet in the graph. Triplets are identified by edge ID which correspond to a unique (head_node, relation_id, tail_node).
- The list of required keys in task configuration file:
    - `description`: task description.
    - `type`: task type (in this case, `EntityLinkPrediction`).
    - `feature`: the attribute(s) used as feature in this task.
    - `train_triplet_set`: the training edge IDs.
    - `val_triplet_set`: the validation edge IDs.
    - `test_triplet_set`: the test edge IDs.

#### `KGRelationPrediction`
- Description: This task requires the model to predict the relation type-id for a triplet in the graph. Triplets are identified by edge ID which correspond to a unique (head_node, relation_id, tail_node).
- The list of required keys in task configuration file:
    - `description`: task description.
    - `type`: task type (in this case, `RelationLinkPrediction`).
    - `feature`: the attribute(s) used as feature in this task.
    - `train_triplet_set`: the training edge IDs.
    - `val_triplet_set`: the validation edge IDs.
    - `test_triplet_set`: the test edge IDs.

## Contributing

We are welcomed to new datasets and tasks! Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md) for more information.