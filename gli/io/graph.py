"""Graph metadata and data saving helper functions."""
import json
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import warnings
from scipy.sparse import spmatrix, coo_matrix, isspmatrix
from .utils import detect_array_type
from ..utils import save_data


class Attribute(object):
    """An attribute of a node, an edge, or a graph."""

    def __init__(self,
                 name,
                 data,
                 description="",
                 data_type=None,
                 data_format=None):
        """
        Initialize the attribute.

        :param name: The name of the attribute.
        :type name: str
        :param data: The data of the attribute.
        :type data: array-like
        :param description: The description of the attribute, defaults to "".
        :type description: str, optional
        :param data_type: The type of the data, which must be one of "int",
            "float", or "str". If not specified, the type will be automatically
            detected, defaults to None.
        :type data_type: str, optional
        :param data_format: The format of the data, which must be one of
            "Tensor" or "SparseTensor". If not specified, the format will be
            automatically detected, defaults to None.
        :type data_format: str, optional

        :raises TypeError: If the input data is not a scipy sparse array or
            numpy array.
        """
        self.name = name
        self.data = data
        self.description = description
        if description == "":
            warnings.warn("The description of the attribute is not specified.")
        self.type = data_type if data_type is not None else detect_array_type(
            data)
        if data_format is not None:
            self.format = data_format
        elif isinstance(data, np.ndarray):
            self.format = "Tensor"
        elif isspmatrix(data):
            self.format = "SparseTensor"
        else:
            raise TypeError("The input data must be a scipy sparse array "
                            "or numpy array.")

        self.num_data = len(data) if self.format == "Tensor" else data.shape[0]

    def get_metadata_dict(self):
        """Return the metadata dictionary of the attribute."""
        return {
            "description": self.description,
            "type": self.type,
            "format": self.format
        }


class UniqueID(Attribute):
    """Unique ID of a node or an edge.

    :param data: The data of the unique id.
    :type data: array
    """

    def __init__(self, data):
        """Initialize the unique id."""
        super().__init__("_ID", data, data_type="int", data_format="Tensor")


class Edges(Attribute):
    """Edges that contain the unique ids of the source and target nodes.

    :param data: The data of the edges.
    :type data: array
    """

    def __init__(self, data):
        """Initialize the edges."""
        super().__init__("_Edge", data, data_type="int", data_format="Tensor")


def save_graph(
    name: str,
    edge: Union[np.ndarray, Dict[Tuple[str, str, str], np.ndarray]],
    num_nodes: Union[int, Dict[str, int], None] = None,
    node_attrs: Union[List[Attribute], Dict[str, List[Attribute]],
                      None] = None,
    edge_attrs: Union[List[Attribute], Dict[Tuple[str, str, str],
                                            List[Attribute]], None] = None,
    graph_node_list: Optional[spmatrix] = None,
    graph_edge_list: Optional[spmatrix] = None,
    graph_attrs: Optional[List[Attribute]] = None,
    is_hetero: bool = False,
    description: str = "",
    cite: str = "",
    save_dir: str = ".",
):
    """Save the graph info by the GLI format.

    This is the recommended way to save a graph (rather than saving manually).
    :func:`save_graph` is the wrapper of :func:`save_homograph` and
    :func:`save_heterograph`. Please see the documentation of these two
    functions for more details about the parameters and usage.
    """
    if not is_hetero:
        return save_homograph(name, edge, num_nodes, node_attrs, edge_attrs,
                              graph_node_list, graph_edge_list, graph_attrs,
                              description, cite, save_dir)
    # verify the inputs are dict for heterograph
    if not isinstance(edge, dict):
        raise TypeError("The input edge must be a dictionary for heterograph.")
    if num_nodes is not None and not isinstance(num_nodes, dict):
        raise TypeError(
            "The input num_nodes must be a dictionary for heterograph.")
    if node_attrs is not None and not isinstance(node_attrs, dict):
        raise TypeError(
            "The input node_attrs must be a dictionary for heterograph..")
    if edge_attrs is not None and not isinstance(edge_attrs, dict):
        raise TypeError(
            "The input edge_attrs must be a dictionary for heterograph..")
    return save_heterograph(name, edge, num_nodes, node_attrs, edge_attrs,
                            graph_node_list, graph_edge_list, graph_attrs,
                            description, cite, save_dir)


def save_homograph(
    name: str,
    edge: np.ndarray,
    num_nodes: Optional[int] = None,
    node_attrs: Optional[List[Attribute]] = None,
    edge_attrs: Optional[List[Attribute]] = None,
    graph_node_list: Optional[spmatrix] = None,
    graph_edge_list: Optional[spmatrix] = None,
    graph_attrs: Optional[List[Attribute]] = None,
    description: str = "",
    citation: str = "",
    save_dir: str = ".",
):
    """
    Save a homogeneous graph information to metadata.json and numpy data files.

    :param name: The name of the graph dataset.
    :type name: str
    :param edge: An array of shape (num_edges, 2). Each row is an edge between
        the two nodes with the given node IDs.
    :type edge: array
    :param num_nodes: The number of nodes in the graph, defaults to None. If
        not specified, the number of nodes will be inferred from ``edge``.
    :type num_nodes: int, optional
    :param node_attrs: A list of attributes of the nodes, defaults to None.
    :type node_attrs: list of Attribute, optional
    :param edge_attrs: A list of attributes of the edges, defaults to None.
    :type edge_attrs: list of Attribute, optional
    :param graph_node_list: An array of shape (num_graphs, num_nodes). Each row
        corresponds to a graph and each column corresponds to a node. The value
        of the element (i, j) is 1 if node j is in graph i, otherwise 0. If not
        specified, the graph will be considered as a single graph, defaults to
        None.
    :type graph_node_list: (sparse) array, optional
    :param graph_edge_list: An array of shape (num_graphs, num_edges). Each
        row corresponds to a graph and each column corresponds to an edge. The
        value of the element (i, j) is 1 if edge j is in graph i, otherwise 0.
        If not specified, the edges contained in each graph specified by
        `graph_node_list` will be considered as all the edges among the nodes
        in the graph, defaults to None.
    :type graph_edge_list: (sparse) array, optional
    :param graph_attrs: A list of attributes of the graphs, defaults to None.
    :type graph_attrs: list of Attribute, optional
    :param description: The description of the dataset, defaults to "".
    :type description: str, optional
    :param citation: The citation of the dataset, defaults to "".
    :type citation: str, optional
    :param save_dir: The directory to save the numpy data files and
        `metadata.json`, defaults to ".".
    :type save_dir: str, optional

    :return: The dictionary of the content in `metadata.json`.
    :rtype: dict

    Example
    -------
    .. code-block:: python

        from gli.io import save_homograph, Attribute
        import numpy as np 
        from numpy.random import randn, randint 
        from scipy.sparse import random as sparse_random

        # Create a graph with 6 nodes and 5 edges.
        edge = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]) # Create attributes of the nodes.
        dense_node_feats = Attribute(
            name="DenseNodeFeature", data=randn(6, 5),  # 6 nodes, 5 features
            description="Dense node features.")
        sparse_node_feats = Attribute(
            name="SparseNodeFeature", data=sparse_random(6, 500),  # 6 nodes, 500 features 
            description="Sparse node features.")
        node_labels = Attribute(
            name="NodeLabel", data=randint(0, 4, 6),  # 6 nodes, 4 classes
            description="Node labels.")
        
        # Save the graph dataset.
        save_homograph(name="example_dataset",
                   edge=edge, node_attrs=[dense_node_feats, sparse_node_feats, node_labels],
                   description="An exampmle dataset.",
                   citation="some bibtex citation")

    The metadata.json and numpy data files will be saved in the current
    directory. And the metadata.json will look like something below.

    .. code-block:: json

        {
            "description": "An exampmle dataset.",
            "data": {
                "Node": {
                    "DenseNodeFeature": {
                        "description": "Dense node features.",
                        "type": "float",
                        "format": "Tensor",
                        "file": "example_dataset__graph__4b7f4a5f08ad24b27423daaa8d445238.npz",
                        "key": "Node_DenseNodeFeature"
                    },
                    "SparseNodeFeature": {
                        "description": "Sparse node features.",
                        "type": "float",
                        "format": "SparseTensor",
                        "file": "example_dataset__graph__Node_SparseNodeFeature__118f9d2bbc457f9d64fe610a9510db1c.sparse.npz"
                    },
                    "NodeLabel": {
                        "description": "Node labels.",
                        "type": "int",
                        "format": "Tensor",
                        "file": "example_dataset__graph__4b7f4a5f08ad24b27423daaa8d445238.npz",
                        "key": "Node_NodeLabel"
                    }
                },
                "Edge": {
                    "_Edge": {
                        "file": "example_dataset__graph__4b7f4a5f08ad24b27423daaa8d445238.npz",
                        "key": "Edge_Edge"
                    }
                },
                "Graph": {
                    "_NodeList": {
                        "file": "example_dataset__graph__4b7f4a5f08ad24b27423daaa8d445238.npz",
                        "key": "Graph_NodeList"
                    }
                }
            },
            "citation": "some bibtex citation",
            "is_heterogeneous": false
        }
    """  # noqa: E501,E262  #pylint: disable=line-too-long
    # Convert attrs to empty lists if they are None.
    if node_attrs is None:
        node_attrs = []
    if edge_attrs is None:
        edge_attrs = []
    if graph_attrs is None:
        graph_attrs = []

    # Check the length of node/edge/graph attrs.
    _verify_attrs(node_attrs, "node")
    _verify_attrs(edge_attrs, "edge")
    _verify_attrs(graph_attrs, "graph")

    # Check `edge` shape.
    if edge.shape[1] != 2:
        raise ValueError("The edge array must have shape (num_edges, 2).")

    # Check the data type and lens of the `graph_node_list` and the
    # `graph_edge_lists`.
    if graph_node_list is None:
        num_nodes = edge.max() + 1 if num_nodes is None else num_nodes
        graph_node_list = coo_matrix(np.ones(num_nodes))
    _verify_graph_lists(graph_node_list, graph_edge_list)

    # Create the data dict to be saved using gli.utils.save_data().
    data = {}
    data["Edge_Edge"] = edge.astype(np.int64)
    for n in node_attrs:
        data[f"Node_{n.name}"] = n.data
    for e in edge_attrs:
        assert e.name != "Edge", "The name of an edge attribute cannot be " \
                                    "'Edge'."
        data[f"Edge_{e.name}"] = e.data
    if graph_node_list is None:
        if len(node_attrs) > 0:
            num_nodes = node_attrs[0].data.shape[0]
        else:
            num_nodes = edge.max() + 1
            # Warning: the number of nodes is not necessarily equal to the
            # maximum node ID in `edge` + 1 when there are isolated nodes.
            warnings.warn("There is no node attributes. The number of nodes "
                          "is inferred from the maximum node ID in `edge` plus"
                          " 1. This may not be correct if there are isolated "
                          "nodes.")
        data["Graph_NodeList"] = np.ones((1, num_nodes), dtype=np.int64)
    else:
        data["Graph_NodeList"] = graph_node_list
    if graph_edge_list is not None:  # _EdgeList is optional in metadata.json.
        data["Graph_EdgeList"] = graph_edge_list
    for g in graph_attrs:
        assert g.name not in ("NodeList", "EdgeList"), \
            "The name of a graph attribute cannot be 'NodeList' or 'EdgeList'."
        data[f"Graph_{g.name}"] = g.data

    # Call save_data().
    key_to_loc = save_data(f"{name}__graph", save_dir=save_dir, **data)

    # Create the metadata dict.
    metadata = {"description": description, "data": {}}

    # Add the metadata of the node attributes.
    node_dict = {}
    for n in node_attrs:
        node_dict[n.name] = _attr_to_metadata_dict(key_to_loc, "Node", n)
    metadata["data"]["Node"] = node_dict

    # Add the metadata of the edge attributes.
    edge_dict = {"_Edge": key_to_loc["Edge_Edge"]}
    for e in edge_attrs:
        edge_dict[e.name] = _attr_to_metadata_dict(key_to_loc, "Edge", e)
    metadata["data"]["Edge"] = edge_dict

    # Add the metadata of the graph attributes.
    graph_dict = {"_NodeList": key_to_loc["Graph_NodeList"]}
    if graph_edge_list is not None:
        graph_dict["_EdgeList"] = key_to_loc["Graph_EdgeList"]
    for g in graph_attrs:
        graph_dict[g.name] = _attr_to_metadata_dict(key_to_loc, "Graph", g)
    metadata["data"]["Graph"] = graph_dict

    metadata["citation"] = citation
    metadata["is_heterogeneous"] = False

    if citation == "":
        warnings.warn("The citation is empty.")

    with open(os.path.join(save_dir, "metadata.json"), "w",
              encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    return metadata


def save_heterograph(
    name: str,
    edge: Dict[Tuple[str, str, str], np.ndarray],
    num_nodes_dict: Optional[Dict[str, int]] = None,
    node_attrs: Optional[Dict[str, List[Attribute]]] = None,
    edge_attrs: Optional[Dict[Tuple[str, str, str], List[Attribute]]] = None,
    graph_node_list: Optional[Dict[str, spmatrix]] = None,
    graph_edge_list: Optional[Dict[Tuple[str, str, str], spmatrix]] = None,
    graph_attrs: Optional[List[Attribute]] = None,
    description: str = "",
    citation: str = "",
    save_dir: str = ".",
):
    """Save a heterogeneous graph info to metadata.json and numpy data files.

    :param name: The name of the graph dataset.
    :type name: str
    :param edge: The key is a tuple of (src_node_type, edge_type,
        dst_node_type). And the map value is a 2D numpy array with shape
        (num_edges, 2). Each row is an edge with the format (src_id, dst_id).
        Each node group should be indexed separately from 0.
    :type edge: Dict[Tuple[str, str, str], array]
    :param num_nodes_dict: The number of nodes in each node group. If None, it will
        be infered from the ``edge``.
    :type num_nodes_dict: Dict[str, int], optional
    :param node_attrs: The node attributes. The key is the node group name and
        the value is a list of Attribute, default to None.
    :type node_attrs: Dict[str, List[Attribute]], optional
    :param edge_attrs: The edge attributes. The key is a tuple of
        (src_node_type, edge_type, dst_node_type) and the value is a list of
        Attribute, default to None.
    :type edge_attrs: Dict[Tuple[str, str, str], List[Attribute]], optional
    :param graph_node_list: A dictionary that maps the node group name to a
        sparse matrix of shape (num_graphs, num_nodes_in_group). Each row
        corresponds to a graph and each column corresponds to a node in that
        node group. The value of the element (i, j) is 1 if node j is in graph
        i, otherwise 0. If not specified, the graph will be considered as a
        single graph, defaults to None.
    :type graph_node_list: Dict[str, spmatrix], optional
    :param graph_edge_list: A dictionary that maps the edge group to a
        sparse matrix of shape (num_graphs, num_edges_in_group). Each row
        corresponds to a graph and each column corresponds to an edge in that
        edge group. The value of the element (i, j) is 1 if edge j is in graph
        i, otherwise 0. If not specified, the graph will be considered as a
        single graph, defaults to None.
    :type graph_edge_list: Dict[Tuple[str, str, str], spmatrix], optional
    :param graph_attrs: The graph attributes, defaults to None.
    :type graph_attrs: List[Attribute], optional
    :param description: The description of the graph dataset, defaults to "".
    :type description: str
    :param citation: The citation of the graph dataset, defaults to "".
        Contributors are strongly encouraged to provide a citation.
    :type citation: str
    :param save_dir: The directory to save the graph dataset, defaults to ".".
    :type save_dir: str
    :return: The dictionary of the content in `metadata.json`.
    :rtype: dict

    Warning
    -------
    Currently gli only support saving a single heterograph dataset. So the
    parameters ``graph_node_list`` and ``graph_edge_list`` are
    essentially redundant. They are only kept for future extension.

    Note
    ----
    Node IDs for each node group should be indexed separately from 0. For
    example, consider a heterogeneous graph with two node groups "user" and
    "item". If there are 3 users and 5 items, the node IDs for "user" should be
    0, 1, 2 and the node IDs for "item" should be 0, 1, 2, 3, 4. gli will
    internally assign a global ID to each node which is unique across all node.
    Users can access the global node ID of a graph `g` by `g.node_map` member.

    Example
    -------
    .. code-block:: python
    
        import numpy as np 
        from numpy.random import randn
        from scipy.sparse import random as sparse_random
        from gli.io import save_heterograph, Attribute
        
        node_groups = ["user", "item"]
        edge_groups = [("user", "click", "item"), ("user", "purchase", "item"),
                       ("user", "is_friend", "user")]
        # Create a sample graph with 3 user nodes and 4+1 item nodes.
        edge = {
            edge_groups[0]: np.array([[0, 0], [0, 1], [1, 2], [2, 3]]),
            edge_groups[1]: np.array([[0, 1], [1, 2]]),
            edge_groups[2]: np.array([[0, 1], [2, 1]])
        }
        
        node_attrs = {
            node_groups[0]: [
                Attribute("UserDenseFeature", randn(3, 5),
                          "Dense user features."),
                Attribute("UserSparseFeature", sparse_random(3, 500),
                          "Sparse user features."),
            ],
            node_groups[1]: [
                Attribute("ItemDenseFeature", randn(5, 5),
                          "Dense item features.")
            ]
        }
        
        edge_attrs = {
            edge_groups[0]: [
                Attribute("ClickTime", randn(4, 1), "Click time.")
            ],
            edge_groups[1]: [
                Attribute("PurchaseTime", randn(2, 1), "Purchase time.")
            ],
            edge_groups[2]: [
                Attribute("SparseFriendFeature", sparse_random(2, 500),
                          "Sparse friend features."),
                Attribute("DenseFriendFeature", randn(2, 5),
                          "Dense friend features.")
            ]
        }
        
        num_nodes_dict = {
            node_groups[0]: 3,
            node_groups[1]:
                5  # more than the actual number of items in the edges
        }
        
        # Save the graph dataset.
        save_heterograph(name="example_hetero_dataset",
                                edge=edge,
                                num_nodes_dict=num_nodes_dict,
                                node_attrs=node_attrs,
                                edge_attrs=edge_attrs,
                                description="An example heterograph dataset.")

    The metadata.json will look like the following:

    .. code-block:: json

        {
            "description": "An example heterograph dataset.",
            "citation": "",
            "data": {
                "Node": {
                    "user": {
                        "UserDenseFeature": {
                            "description": "Dense user features.",
                            "type": "float",
                            "format": "Tensor",
                            "file": "example_hetero_dataset__heterograph__aab19db19513942e161ace237aea63b4.npz",
                            "key": "Node_user_UserDenseFeature"
                        },
                        "UserSparseFeature": {
                            "description": "Sparse user features.",
                            "type": "float",
                            "format": "SparseTensor",
                            "file": "example_hetero_dataset__heterograph__Node_user_UserSparseFeature__30209d631dcc4ae3813d3c360f9c42dd.sparse.npz"
                        },
                        "_ID": {
                            "description": "",
                            "type": "int",
                            "format": "Tensor",
                            "file": "example_hetero_dataset__heterograph__aab19db19513942e161ace237aea63b4.npz",
                            "key": "Node_user__ID"
                        }
                    },
                    "item": {
                        "ItemDenseFeature": {
                            "description": "Dense item features.",
                            "type": "float",
                            "format": "Tensor",
                            "file": "example_hetero_dataset__heterograph__aab19db19513942e161ace237aea63b4.npz",
                            "key": "Node_item_ItemDenseFeature"
                        },
                        "_ID": {
                            "description": "",
                            "type": "int",
                            "format": "Tensor",
                            "file": "example_hetero_dataset__heterograph__aab19db19513942e161ace237aea63b4.npz",
                            "key": "Node_item__ID"
                        }
                    }
                },
                "Edge": {
                    "user_click_item": {
                        "ClickTime": {
                            "description": "Click time.",
                            "type": "float",
                            "format": "Tensor",
                            "file": "example_hetero_dataset__heterograph__aab19db19513942e161ace237aea63b4.npz",
                            "key": "Edge_user_click_item_ClickTime"
                        },
                        "_ID": {
                            "description": "",
                            "type": "int",
                            "format": "Tensor",
                            "file": "example_hetero_dataset__heterograph__aab19db19513942e161ace237aea63b4.npz",
                            "key": "Edge_user_click_item__ID"
                        },
                        "_Edge": {
                            "description": "",
                            "type": "int",
                            "format": "Tensor",
                            "file": "example_hetero_dataset__heterograph__aab19db19513942e161ace237aea63b4.npz",
                            "key": "Edge_user_click_item__Edge"
                        }
                    },
                    "user_purchase_item": {
                        "PurchaseTime": {
                            "description": "Purchase time.",
                            "type": "float",
                            "format": "Tensor",
                            "file": "example_hetero_dataset__heterograph__aab19db19513942e161ace237aea63b4.npz",
                            "key": "Edge_user_purchase_item_PurchaseTime"
                        },
                        "_ID": {
                            "description": "",
                            "type": "int",
                            "format": "Tensor",
                            "file": "example_hetero_dataset__heterograph__aab19db19513942e161ace237aea63b4.npz",
                            "key": "Edge_user_purchase_item__ID"
                        },
                        "_Edge": {
                            "description": "",
                            "type": "int",
                            "format": "Tensor",
                            "file": "example_hetero_dataset__heterograph__aab19db19513942e161ace237aea63b4.npz",
                            "key": "Edge_user_purchase_item__Edge"
                        }
                    },
                    "user_is_friend_user": {
                        "SparseFriendFeature": {
                            "description": "Sparse friend features.",
                            "type": "float",
                            "format": "SparseTensor",
                            "file": "example_hetero_dataset__heterograph__Edge_user_is_friend_user_SparseFriendFeature__fc3b5ebfe3efe6ac35e116c02d388ac6.sparse.npz"
                        },
                        "DenseFriendFeature": {
                            "description": "Dense friend features.",
                            "type": "float",
                            "format": "Tensor",
                            "file": "example_hetero_dataset__heterograph__aab19db19513942e161ace237aea63b4.npz",
                            "key": "Edge_user_is_friend_user_DenseFriendFeature"
                        },
                        "_ID": {
                            "description": "",
                            "type": "int",
                            "format": "Tensor",
                            "file": "example_hetero_dataset__heterograph__aab19db19513942e161ace237aea63b4.npz",
                            "key": "Edge_user_is_friend_user__ID"
                        },
                        "_Edge": {
                            "description": "",
                            "type": "int",
                            "format": "Tensor",
                            "file": "example_hetero_dataset__heterograph__aab19db19513942e161ace237aea63b4.npz",
                            "key": "Edge_user_is_friend_user__Edge"
                        }
                    }
                },
                "Graph": {
                    "_NodeList": {
                        "file": "example_hetero_dataset__heterograph__Graph_NodeList__752140b0bd5669a2580f06dda6a70ced.sparse.npz"
                    }
                }
            },
            "is_heterogeneous": true
        }
    """  # noqa: E501,E262  #pylint: disable=line-too-long
    _verify_hetero_type(edge, node_attrs, edge_attrs, graph_attrs)

    edge_groups = list(edge.keys())
    if num_nodes_dict is None:
        # infer the node groups from edge groups
        node_groups = set()
        for src, rel, dst in edge_groups:
            node_groups.add(src)
            node_groups.add(dst)
        node_groups = list(node_groups)
        num_nodes_dict = _infer_num_nodes_dict(edge, node_groups)
    else:
        node_groups = list(num_nodes_dict.keys())

    # Convert attrs to default empty val if they are None.
    if node_attrs is None:
        node_attrs = {node_group: [] for node_group in node_groups}
    if edge_attrs is None:
        edge_attrs = {edge_group: [] for edge_group in edge_groups}
    if graph_attrs is None:
        graph_attrs = []

    # Check the length of the node groups and node attributes.
    for node_group in node_groups:
        _verify_attrs(node_attrs[node_group], f"node/{node_group}")
    for edge_group in edge_groups:
        _verify_attrs(edge_attrs[edge_group], f"edge/{edge_group}")
    _verify_attrs(graph_attrs, "graph")

    # Check the data type and lens of the `graph_node_list` and
    # `graph_edge_list`.
    if graph_node_list is None:
        num_all_nodes = sum(num_nodes_dict.values())
        graph_node_list = coo_matrix(np.ones(num_all_nodes, dtype=np.int64))
    _verify_graph_lists(graph_node_list, graph_edge_list)
    # check if the graph_node_list and graph_edge_list contains multi graphs
    if graph_node_list.shape[0] > 1 or (graph_edge_list is not None
                                        and graph_edge_list.shape[0] > 1):
        raise NotImplementedError(
            "Currently, gli only supports saving a single hetero graph.")

    # Assign unique IDs to the node groups and edge groups.
    node_maps, edge_maps = _assign_id(edge, num_nodes_dict)

    # Add the unique IDs as attributes.
    for node_group in node_groups:
        node_attrs[node_group].append(UniqueID(node_maps[node_group]))
    for edge_group in edge_groups:
        edge_attrs[edge_group].append(UniqueID(edge_maps[edge_group]))

    # Create the edge array with new IDs.
    new_edges = {}
    for edge_group in edge_groups:
        src_node_group, rel, dst_node_group = edge_group
        src_node_map = node_maps[src_node_group]
        dst_node_map = node_maps[dst_node_group]
        group_edge_array = edge[edge_group]
        new_edges[edge_group] = np.stack([
            src_node_map[group_edge_array[:, 0]],
            dst_node_map[group_edge_array[:, 1]]
        ],
                                         axis=1)  # shape (E, 2)
    for edge_group in edge_groups:
        edge_attrs[edge_group].append(Edges(new_edges[edge_group]))

    data = {}
    for node_group in node_groups:
        for attr in node_attrs[node_group]:
            key = f"Node_{node_group}_{attr.name}"
            data[key] = attr.data
    for edge_group in edge_groups:
        src_node_group = edge_group[0]
        rel = edge_group[1]
        dst_node_group = edge_group[2]
        for attr in edge_attrs[edge_group]:
            key = f"Edge_{src_node_group}_{rel}_{dst_node_group}_{attr.name}"
            data[key] = attr.data
    for attr in graph_attrs:
        key = f"Graph_{attr.name}"
        data[key] = attr.data
    data["Graph_NodeList"] = graph_node_list
    if graph_edge_list is not None:
        data["Graph_EdgeList"] = graph_edge_list

    key_to_loc = save_data(f"{name}__heterograph", save_dir=save_dir, **data)

    # Create the metadata dict.
    metadata = {
        "description": description,
        "citation": citation,
        "data": {},
        "is_heterogeneous": True
    }
    node_dict = {}
    for node_group in node_groups:
        node_dict[node_group] = {}
        for attr in node_attrs[node_group]:
            node_dict[node_group][attr.name] = _attr_to_metadata_dict(
                key_to_loc, f"Node_{node_group}", attr)
    metadata["data"]["Node"] = node_dict
    edge_dict = {}
    for edge_group in edge_groups:
        src_node_group = edge_group[0]
        rel = edge_group[1]
        dst_node_group = edge_group[2]
        edge_group_key = f"{src_node_group}_{rel}_{dst_node_group}"
        edge_dict[edge_group_key] = {}
        for attr in edge_attrs[edge_group]:
            edge_dict[edge_group_key][attr.name] = _attr_to_metadata_dict(
                key_to_loc, f"Edge_{src_node_group}_{rel}_{dst_node_group}",
                attr)
    metadata["data"]["Edge"] = edge_dict
    graph_dict = {"_NodeList": key_to_loc["Graph_NodeList"]}
    if graph_edge_list is not None:
        graph_dict["_EdgeList"] = key_to_loc["Graph_EdgeList"]
    for attr in graph_attrs:
        graph_dict[attr.name] = _attr_to_metadata_dict(key_to_loc, "Graph",
                                                       attr)
    metadata["data"]["Graph"] = graph_dict

    if citation == "":
        warnings.warn("The citation is empty.")

    with open(os.path.join(save_dir, "metadata.json"), "w",
              encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    return metadata


def _verify_hetero_type(
    edge: Dict[Tuple[str, str, str], np.ndarray],
    node_attrs: Optional[Dict[str, List[Attribute]]] = None,
    edge_attrs: Optional[Dict[Tuple[str, str, str], List[Attribute]]] = None,
    graph_attrs: Optional[List[Attribute]] = None,
):
    if not isinstance(edge, dict):
        raise TypeError(
            "The `edge` must be a dict of (str, str) -> np.ndarray.")
    assert all(isinstance(k, tuple) and len(k) == 3
               for k in edge), "The keys of `edge` must be tuples of length 3."
    assert all(
        isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 2
        for v in edge.values()
    ), "values of `edge` must be 2D numpy arrays with shape (num_edges, 2)."
    if node_attrs is not None and not isinstance(node_attrs, dict):
        raise TypeError(
            "The `node_attrs` must be a dict of str -> List[Attribute].")
    if edge_attrs is not None and not isinstance(edge_attrs, dict):
        raise TypeError("`edge_attrs` must be a dict"
                        "of (str, str, str) -> List[Attribute].")
    if graph_attrs is not None and not isinstance(graph_attrs, list):
        raise TypeError("The `graph_attrs` must be a list of Attribute.")


def _verify_attrs(attrs, object_name):
    """Verify all elements in attrs are proper inputs.

    In particular, test if all elements in attrs are Attribute and have the
    same length of data.
    """
    if len(attrs) > 0:
        if not (isinstance(attrs, list)
                and all(isinstance(a, Attribute) for a in attrs)):
            raise TypeError(f"All elements of {object_name}_attrs must be "
                            "Attribute.")
        num_data = attrs[0].num_data
        for attr in attrs:
            if attr.num_data != num_data:
                raise ValueError("the length of data of all attributes of "
                                 f"the {object_name}(s) must be the same.")


def _verify_graph_lists(graph_node_list: spmatrix, graph_edge_list: spmatrix):
    """Verify the graph_node_list and graph_edge_list.

    Criteria:
    1. graph_node_list and graph_edge_list must be scipy sparse matrices.
    2. graph_node_list and graph_edge_list must have the same length.
    3. graph_node_list and graph_edge_list must have the dtype of bool or
    int64 (only contains 0 and 1).
    """

    def _verify_single_graph_list(graph_list, object_name):
        if graph_list is None:
            return
        if not isspmatrix(graph_list):
            raise TypeError(f"{object_name} must be a scipy sparse matrix.")
        if graph_list.dtype.name == "bool":
            return
        if graph_list.dtype.name == "int64":
            # check if data contains only 0 and 1
            unique_values = np.unique(graph_list.data)
            assert len(unique_values) == 1 and unique_values[0] == 1, \
                f"data of {object_name} must contain only 0 and 1."

    if graph_node_list is None:
        if graph_edge_list is None:
            return
        else:
            raise ValueError("graph_node_list must be specified if "
                             "graph_edge_list is specified.")
    else:
        if not isspmatrix(graph_node_list):
            raise TypeError("graph_node_list must be a scipy sparse matrix.")
        _verify_single_graph_list(graph_node_list, "graph_node_list")
        if graph_edge_list is None:
            return
        _verify_single_graph_list(graph_edge_list, "graph_edge_list")
        assert graph_node_list.shape[0] == graph_edge_list.shape[0], \
            "graph_node_list and graph_edge_list must have the same."


def _attr_to_metadata_dict(key_to_loc, prefix, a):
    """Obtain the metadata dict of the attribute.

    Args:
        prefix (str): The prefix of the attribute, i.e., "Node", "Edge", or
            "Graph".
        a (Attribute): The attribute.
    Returns:
        dict: The metadata dict of the attribute.
    """
    metadata = {}
    metadata.update(a.get_metadata_dict())
    metadata.update(key_to_loc[f"{prefix}_{a.name}"])
    return metadata


def _assign_id(edge: Dict[Tuple[str, str, str], np.ndarray],
               num_nodes_dict: Dict[str, int]):
    """Assign unique IDs to the node groups and edge groups."""
    node_maps = {}  # original ID -> new ID
    edge_maps = {}  # original ID -> new ID

    offset = 0
    for node_group in sorted(num_nodes_dict.keys()):
        node_maps[node_group] = np.arange(num_nodes_dict[node_group],
                                          dtype=np.int64) + offset
        offset += num_nodes_dict[node_group]

    offset = 0
    for edge_group in sorted(edge.keys()):
        edge_maps[edge_group] = np.arange(len(edge[edge_group]),
                                          dtype=np.int64) + offset
        offset += len(edge[edge_group])

    return node_maps, edge_maps


def _infer_num_nodes_dict(edge: Dict[Tuple[str, str, str], np.ndarray],
                          node_groups: List[str]):
    """Infer the number of nodes in each node group from the edge list."""
    num_nodes_dict = {node_group: 0 for node_group in node_groups}
    for (src_group, dst_group), edge_array in edge.items():
        num_nodes_dict[src_group] = max(num_nodes_dict[src_group],
                                        edge_array[:, 0].max() + 1)
        num_nodes_dict[dst_group] = max(num_nodes_dict[dst_group],
                                        edge_array[:, 1].max() + 1)
    return num_nodes_dict
