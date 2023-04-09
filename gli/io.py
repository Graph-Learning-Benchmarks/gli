"""Helper functions for creating datasets in GLI format."""
import json
import os
import warnings
import numpy as np
from scipy.sparse import isspmatrix

from gli.utils import save_data


def detect_array_type(array):
    """
    Detect the type of the data in the array.

    :param array: The input array.
    :type array: scipy.sparse array or numpy array

    :raises ValueError: If the input array is empty.
    :raises TypeError: If the input array is not a scipy sparse array or numpy
        array.
    :raises TypeError: If the input array contains unsupported data types.

    :return: Data type of the elements in the array.
    :rtype: str
    """
    if isspmatrix(array) or isinstance(array, np.ndarray):
        if array.size == 0:
            raise ValueError("The input array is empty.")

        # Check for the first non-null element's type
        if isspmatrix(array):
            data = array.data
        else:
            # In case array is a multi-dimensional numpy array, flatten it
            # Otherwise we will not be able to iterate data in the for loop
            data = array.flatten().data
        for element in data:
            if element is not None:
                element_type = type(element)
                break

        if issubclass(element_type, int):
            return "int"
        elif issubclass(element_type, float) or element_type is np.float32:
            # np.float64 is a subclass of float but np.float32 is not
            return "float"
        elif issubclass(element_type, str):
            return "str"
        else:
            raise TypeError("The input array contains unsupported data types.")
    else:
        raise TypeError("The input array must be a scipy sparse array or numpy"
                        " array.")


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
        self.num_data = len(data)
        self.description = description
        if description == "":
            warnings.warn("The description of the attribute is not specified.")
        if data_type is not None:
            self.type = data_type
        else:
            self.type = detect_array_type(data)
        if data_format is not None:
            self.data_format = data_format
        else:
            if isinstance(data, np.ndarray):
                self.format = "Tensor"
            elif isspmatrix(data):
                self.format = "SparseTensor"
            else:
                raise TypeError("The input data must be a scipy sparse array "
                                "or numpy array.")

    def get_metadata_dict(self):
        """Return the metadata dictionary of the attribute."""
        return {
            "description": self.description,
            "type": self.type,
            "format": self.format
        }


def save_graph(name,
               edge,
               node_attrs=None,
               edge_attrs=None,
               graph_node_lists=None,
               graph_edge_lists=None,
               graph_attrs=None,
               description="",
               citation="",
               is_heterogeneous=False,
               save_dir="."):
    """
    Save the graph information to metadata.json and numpy data files.

    :param name: The name of the graph dataset.
    :type name: str
    :param edge: An array of shape (num_edges, 2). Each row is an edge between
        the two nodes with the given node IDs.
    :type edge: array
    :param node_attrs: A list of attributes of the nodes, defaults to None.
    :type node_attrs: list of Attribute, optional
    :param edge_attrs: A list of attributes of the edges, defaults to None.
    :type edge_attrs: list of Attribute, optional
    :param graph_node_lists: An array of shape (num_graphs, num_nodes). Each
        row corresponds to a graph and each column corresponds to a node. The
        value of the element (i, j) is 1 if node j is in graph i, otherwise 0.
        If not specified, the graph will be considered as a single graph,
        defaults to None.
    :type graph_node_lists: (sparse) array, optional
    :param graph_edge_lists: An array of shape (num_graphs, num_edges). Each
        row corresponds to a graph and each column corresponds to an edge. The
        value of the element (i, j) is 1 if edge j is in graph i, otherwise 0.
        If not specified, the edges contained in each graph specified by
        `graph_node_lists` will be considered as all the edges among the nodes
        in the graph, defaults to None.
    :type graph_edge_lists: (sparse) array, optional
    :param graph_attrs: A list of attributes of the graphs, defaults to None.
    :type graph_attrs: list of Attribute, optional
    :param description: The description of the dataset, defaults to "".
    :type description: str, optional
    :param citation: The citation of the dataset, defaults to "".
    :type citation: str, optional
    :param is_heterogeneous: Whether the graph is heterogeneous, defaults to
        False.
    :type is_heterogeneous: bool, optional
    :param save_dir: The directory to save the numpy data files and
        `metadata.json`, defaults to ".".
    :type save_dir: str, optional

    :raises ValueError: If the length of data of all attributes of the node(s)
        or edge(s) or graph(s) is not the same.
    :raises ValueError: If the edge array does not have shape (num_edges, 2).
    :raises ValueError: If the data type of the `graph_node_lists` and the
        `graph_edge_lists` are not binary.
    :raises ValueError: If the number of graphs in the `graph_node_lists` and
        the `graph_edge_lists` are not the same.
    :raises NotImplementedError: If the graph is heterogeneous.

    :return: The dictionary of the content in `metadata.json`.
    :rtype: dict

    Example
    -------
    .. code-block:: python

        import numpy as np
        from numpy.random import randn, randint
        from scipy.sparse import random as sparse_random

        # Create a graph with 6 nodes and 5 edges.
        edge = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
        # Create attributes of the nodes.
        dense_node_feats = Attribute(
            name="DenseNodeFeature",
            data=randn(6, 5),  # 6 nodes, 5 features
            description="Dense node features.")
        sparse_node_feats = Attribute(
            name="SparseNodeFeature",
            data=sparse_random(6, 500),  # 6 nodes, 500 features
            description="Sparse node features.")
        node_labels = Attribute(
            name="NodeLabel",
            data=randint(0, 4, 6),  # 6 nodes, 4 classes
            description="Node labels.")

        # Save the graph dataset.
        save_graph(name="example_dataset",
                   edge=edge,
                   node_attrs=[dense_node_feats, sparse_node_feats, node_labels],
                   description="An exampmle dataset.",
                   citation="some bibtex citation")

        # The metadata.json and numpy data files will be saved in the current
        # directory. And the metadata.json will look like something below.

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

    def _verify_attrs_length(attrs, object_name):
        """Verify all elements in attrs have the same length."""
        if len(attrs) > 0:
            num_data = attrs[0].num_data
            for attr in attrs:
                if attr.num_data != num_data:
                    raise ValueError("The length of data of all attributes of "
                                     f"the {object_name}(s) must be the same.")

    # Check the length of node/edge/graph attrs.
    _verify_attrs_length(node_attrs, "node")
    _verify_attrs_length(edge_attrs, "edge")
    _verify_attrs_length(graph_attrs, "graph")

    # Check `edge` shape.
    if edge.shape[1] != 2:
        raise ValueError("The edge array must have shape (num_edges, 2).")

    # Check the data type of `graph_node_lists` and `graph_edge_lists` are
    # either boolean or integer with only 0 and 1.
    if graph_node_lists is not None:
        if graph_node_lists.dtype != np.bool and \
                (graph_node_lists.dtype != np.int32 or
                 np.any(graph_node_lists != 0) and
                 np.any(graph_node_lists != 1)):
            raise ValueError("The data type of `graph_node_lists` must be "
                             "either boolean or integer with only 0 and 1.")
    if graph_edge_lists is not None:
        if graph_edge_lists.dtype != np.bool and \
                (graph_edge_lists.dtype != np.int32 or
                 np.any(graph_edge_lists != 0) and
                 np.any(graph_edge_lists != 1)):
            raise ValueError("The data type of `graph_edge_lists` must be "
                             "either boolean or integer with only 0 and 1.")

    # Check the number of graphs in `graph_node_lists` and `graph_edge_lists`
    # are equal.
    if graph_node_lists is not None and graph_edge_lists is not None:
        if graph_node_lists.shape[0] != graph_edge_lists.shape[0]:
            raise ValueError("The number of graphs in the `graph_node_lists` "
                             "must be equal to the number of graphs in the "
                             "`graph_edge_lists`.")

    # Create the data dict to be saved using gli.utils.save_data().
    data = {}
    data["Edge_Edge"] = edge.astype(np.int32)
    for n in node_attrs:
        data[f"Node_{n.name}"] = n.data
    for e in edge_attrs:
        assert e.name != "Edge", "The name of an edge attribute cannot be " \
                                    "'Edge'."
        data[f"Edge_{e.name}"] = e.data
    if graph_node_lists is None:
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
        data["Graph_NodeList"] = np.ones((1, num_nodes), dtype=np.int32)
    else:
        data["Graph_NodeList"] = graph_node_lists
    if graph_edge_lists is not None:  # _EdgeList is optional in metadata.json.
        data["Graph_EdgeList"] = graph_edge_lists
    for g in graph_attrs:
        assert g.name not in ("NodeList", "EdgeList"), \
            "The name of a graph attribute cannot be 'NodeList' or 'EdgeList'."
        data[f"Graph_{g.name}"] = g.data
    # Call save_data().
    key_to_loc = save_data(f"{name}__graph", save_dir=save_dir, **data)

    def _attr_to_metadata_dict(prefix, a):
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

    # Create the metadata dict.
    metadata = {"description": description, "data": {}}
    # Add the metadata of the node attributes.
    node_dict = {}
    for n in node_attrs:
        node_dict[n.name] = _attr_to_metadata_dict("Node", n)
    metadata["data"]["Node"] = node_dict
    # Add the metadata of the edge attributes.
    edge_dict = {"_Edge": key_to_loc["Edge_Edge"]}
    for e in edge_attrs:
        edge_dict[e.name] = _attr_to_metadata_dict("Edge", e)
    metadata["data"]["Edge"] = edge_dict
    # Add the metadata of the graph attributes.
    graph_dict = {"_NodeList": key_to_loc["Graph_NodeList"]}
    if graph_edge_lists is not None:
        graph_dict["_EdgeList"] = key_to_loc["Graph_EdgeList"]
    for g in graph_attrs:
        graph_dict[g.name] = _attr_to_metadata_dict("Graph", g)
    metadata["data"]["Graph"] = graph_dict

    metadata["citation"] = citation
    metadata["is_heterogeneous"] = is_heterogeneous

    if citation == "":
        warnings.warn("The citation is empty.")
    if is_heterogeneous:
        raise NotImplementedError(
            "Heterogeneous graphs are not supported yet.")

    with open(os.path.join(save_dir, "metadata.json"), "w",
              encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    return metadata


def _check_data_splits(train_set, val_set, test_set, train_ratio, val_ratio,
                       test_ratio, num_samples):
    """Check the input arguments of data splits are valid."""
    if train_set is None:
        assert num_samples is not None, \
            "If `train_set` is not provided, `num_samples` must be provided."
        assert 0 < train_ratio + val_ratio + test_ratio <= 1, \
            "The sum of `train_ratio`, `val_ratio`, and `test_ratio` must " \
            "be in (0, 1]."
    else:
        assert val_set is not None and test_set is not None, \
            "If `train_set` is provided, `val_set` and `test_set` must be " \
            "provided."
        if isinstance(train_set, np.ndarray):
            train_set = train_set.tolist()
        if isinstance(val_set, np.ndarray):
            val_set = val_set.tolist()
        if isinstance(test_set, np.ndarray):
            test_set = test_set.tolist()
        assert isinstance(train_set, list) and \
            isinstance(val_set, list) and isinstance(test_set, list), \
            "`train_set`, `val_set`, and `test_set` must be lists or numpy" \
            " arrays."
        if isinstance(train_set[0], list):  # Multiple splits.
            assert len(train_set) == len(val_set) == len(test_set), \
                "If `train_set`, `val_set`, and `test_set` are lists of " \
                "lists, they must have the same length (number of splits)."
            # Check the split ratio's of different splits are the same.
            split_num = (len(train_set[0]), len(val_set[0]), len(test_set[0]))
            for i in range(len(train_set)):
                assert len(train_set[i]) == split_num[0] and \
                    len(val_set[i]) == split_num[1] and \
                    len(test_set[i]) == split_num[2], \
                    "The split ratio's of different splits are not the same."

    # Return `train_set`, `val_set`, and `test_set` as lists of lists.
    return train_set, val_set, test_set


def _check_feature(feature):
    """Check the input argument `feature` is valid."""
    assert isinstance(feature, list), \
        "`feature` must be a list of strings."
    for feat in feature:
        assert isinstance(feat, str), \
            "Each element in `feature` must be a string."
        assert feat.startswith("Node/") or feat.startswith("Edge/") or \
            feat.startswith("Graph/"), \
            "Each element in `feature` must be a node/edge/graph attribute."


def _check_node_level_target(target):
    """Check the input argument `target` for node level tasks is valid."""
    assert isinstance(target, str), \
        "`target` must be a string."
    assert target.startswith("Node/"), \
        "`target` must be a node attribute."


def save_task_node_regression(name,
                              description,
                              feature,
                              target,
                              train_set=None,
                              val_set=None,
                              test_set=None,
                              train_ratio=0.8,
                              val_ratio=0.1,
                              test_ratio=0.1,
                              num_samples=None,
                              task_id=1,
                              save_dir="."):
    """Save the node regression task information into task json and data files.

    :param name: The name of the dataset.
    :type name: str
    :param description: The description of the task.
    :type description: str
    :param feature: The list of feature names to be used in the task. The
        features could be node attributes, edge attributes, or graph
        attributes. For homogeneous graphs, the feature names should be in the
        format of "Node/{node_attr_name}", "Edge/{edge_attr_name}", or
        "Graph/{graph_attr_name}". For heterogeneous graphs, the feature names
        should be in the format of "Node/{node_type}/{node_attr_name}",
        "Edge/{edge_type}/{edge_attr_name}", or
        "Graph/{graph_type}/{graph_attr_name}". The node/edge/graph_attr_name,
        and the node/edge/graph_type should be the ones declared in the
        metadata.json file.
    :type feature: list of str
    :param target: The attribute name as prediction target in the task. For
        this node regression task, the attribute should be a node attribute.
        For homogeneous graphs, the attribute name should be in the format of
        "Node/{node_attr_name}". For heterogeneous graphs, the attribute name
        should be in the format of "Node/{node_type}/{node_attr_name}". The
        node_attr_name and the node_type should be the ones declared in the
        metadata.json file.
    :type target: str
    :param train_set: A list of training node IDs or a list of list training
        node IDs. In the latter case, each inner list is the node IDs of one
        data split. If not None, `train_ratio`, `val_ratio`, and `test_ratio`
        will be ignored while `val_set` and `test_set` must present. If None,
        the task json file will store `train_ratio`, `val_ratio`, and
        `test_ratio` and random splits will be generated at run time. Default:
        None.
    :type train_set: list/array of int or list of lists/2-d array of int
    :param val_set: A list of validation node IDs or a list of list validation
        node IDs. See `train_set` for more details. Default: None.
    :type val_set: list/array of int or list of lists/2-d array of int
    :param test_set: A list of test node IDs or a list of list test node IDs.
        See `train_set` for more details. Default: None.
    :type test_set: list/array of int or list of lists/2-d array of int
    :param train_ratio: The ratio of training nodes. See `train_set` for more
        details. Default: 0.8.
    :type train_ratio: float
    :param val_ratio: The ratio of validation nodes. See `train_set` for more
        details. Default: 0.1.
    :type val_ratio: float
    :param test_ratio: The ratio of test nodes. See `train_set` for more
        details. Default: 0.1.
    :type test_ratio: float
    :param num_samples: The total number of nodes in the dataset. This needs to
        be provided if `train_set`, `val_set`, and `test_set` are not provided.
        Default: None.
    :type num_samples: int
    :param task_id: The task ID. This is needed when there are multiple tasks
        of the same task type are defined on the dataset. Default: 1.
    :type task_id: int
    :param save_dir: The directory to save the task json and data files.
        Default: ".".
    :type save_dir: str

    Example
    -------
    .. code-block:: python

        train_set = [0, 1]
        val_set = [2, 3]
        test_set = [4, 5]

        # Save the task information.
        save_task_node_regression(
            name="example_dataset",
            description="A node regression task for the example dataset.",
            feature=["Node/DenseNodeFeature", "Node/SparseNodeFeature"],
            target="Node/NodeLabel",
            train_set=train_set,
            val_set=val_set,
            test_set=test_set)
        # This function will save the task information into a json file named
        # `task_node_regression_1.json` and one numpy data file storing the
        # data splits, `train_set`, `val_set`, and `test_set`. The json file
        # will look like the following.

    .. code-block:: json

        {
            "description": "A node regression task for the example dataset.",
            "type": "NodeRegression",
            "feature": [
                "Node/DenseNodeFeature",
                "Node/SparseNodeFeature"
            ],
            "target": "Node/NodeLabel",
            "train_set": {
                "file": "example_dataset__task_node_classification_1__4dcac617700f69a6dec06c2b5f75a246.npz",
                "key": "train_set"
            },
            "val_set": {
                "file": "example_dataset__task_node_classification_1__4dcac617700f69a6dec06c2b5f75a246.npz",
                "key": "val_set"
            },
            "test_set": {
                "file": "example_dataset__task_node_classification_1__4dcac617700f69a6dec06c2b5f75a246.npz",
                "key": "test_set"
            }
        }
    """  # noqa: E501,E262  #pylint: disable=line-too-long
    # Check the input arguments.
    assert isinstance(description, str), \
        "`description` must be a string."
    _check_feature(feature)
    _check_node_level_target(target)
    train_set, val_set, test_set = _check_data_splits(train_set, val_set,
                                                      test_set, train_ratio,
                                                      val_ratio, test_ratio,
                                                      num_samples)

    # Create the dictionary for task json file.
    task_dict = {
        "description": description,
        "type": "NodeRegression",
        "feature": feature,
        "target": target
    }
    if train_set is not None:
        # Save the task data files, i.e., the data splits in this task.
        data_dict = {
            "train_set": np.array(train_set),
            "val_set": np.array(val_set),
            "test_set": np.array(test_set)
        }
        key_to_loc = save_data(f"{name}__task_node_classification_{task_id}",
                               save_dir=save_dir,
                               **data_dict)
        # Update the task dictionary with the data file names and keys.
        task_dict.update(key_to_loc)
        if isinstance(train_set[0], list):
            task_dict["num_splits"] = len(train_set)
    else:
        task_dict["train_ratio"] = train_ratio
        task_dict["val_ratio"] = val_ratio
        task_dict["test_ratio"] = test_ratio
        task_dict["num_samples"] = num_samples

    # Save the task json file.
    with open(os.path.join(save_dir, f"task_node_regression_{task_id}.json"),
              "w",
              encoding="utf-8") as f:
        json.dump(task_dict, f, indent=4)

    return task_dict


def save_task_node_classification(name,
                                  description,
                                  feature,
                                  target,
                                  num_classes,
                                  train_set=None,
                                  val_set=None,
                                  test_set=None,
                                  train_ratio=0.8,
                                  val_ratio=0.1,
                                  test_ratio=0.1,
                                  num_samples=None,
                                  task_id=1,
                                  save_dir="."):
    """Save the node classification task information into task json and data files.

    :param name: The name of the dataset.
    :type name: str
    :param description: The description of the task.
    :type description: str
    :param feature: The list of feature names to be used in the task. The
        features could be node attributes, edge attributes, or graph
        attributes. For homogeneous graphs, the feature names should be in the
        format of "Node/{node_attr_name}", "Edge/{edge_attr_name}", or
        "Graph/{graph_attr_name}". For heterogeneous graphs, the feature names
        should be in the format of "Node/{node_type}/{node_attr_name}",
        "Edge/{edge_type}/{edge_attr_name}", or
        "Graph/{graph_type}/{graph_attr_name}". The node/edge/graph_attr_name,
        and the node/edge/graph_type should be the ones declared in the
        metadata.json file.
    :type feature: list of str
    :param target: The attribute name as prediction target in the task. For
        a node classification task, the attribute should be a node attribute.
        For homogeneous graphs, the attribute name should be in the format of
        "Node/{node_attr_name}". For heterogeneous graphs, the attribute name
        should be in the format of "Node/{node_type}/{node_attr_name}". The
        node_attr_name and the node_type should be the ones declared in the
        metadata.json file.
    :type target: str
    :param num_classes: The number of classes in the task.
    :type num_classes: int
    :param train_set: A list of training node IDs or a list of list training
        node IDs. In the latter case, each inner list is the node IDs of one
        data split. If not None, `train_ratio`, `val_ratio`, and `test_ratio`
        will be ignored while `val_set` and `test_set` must present. If None,
        the task json file will store `train_ratio`, `val_ratio`, and
        `test_ratio` and random splits will be generated at run time. Default:
        None.
    :type train_set: list/array of int or list of lists/2-d array of int
    :param val_set: A list of validation node IDs or a list of list validation
        node IDs. See `train_set` for more details. Default: None.
    :type val_set: list/array of int or list of lists/2-d array of int
    :param test_set: A list of test node IDs or a list of list test node IDs.
        See `train_set` for more details. Default: None.
    :type test_set: list/array of int or list of lists/2-d array of int
    :param train_ratio: The ratio of training nodes. See `train_set` for more
        details. Default: 0.8.
    :type train_ratio: float
    :param val_ratio: The ratio of validation nodes. See `train_set` for more
        details. Default: 0.1.
    :type val_ratio: float
    :param test_ratio: The ratio of test nodes. See `train_set` for more
        details. Default: 0.1.
    :type test_ratio: float
    :param num_samples: The total number of nodes in the dataset. This needs to
        be provided if `train_set`, `val_set`, and `test_set` are not provided.
        Default: None.
    :type num_samples: int
    :param task_id: The task ID. This is needed when there are multiple tasks
        of the same task type are defined on the dataset. Default: 1.
    :type task_id: int
    :param save_dir: The directory to save the task json and data files.
        Default: ".".
    :type save_dir: str

    Example
    -------
    .. code-block:: python

        train_set = [0, 1]
        val_set = [2, 3]
        test_set = [4, 5]

        # Save the task information.
        save_task_node_classification(
            name="example_dataset",
            description="A node classification task for the example dataset.",
            feature=["Node/DenseNodeFeature", "Node/SparseNodeFeature"],
            target="Node/NodeLabel",
            num_classes=4,
            train_set=train_set,
            val_set=val_set,
            test_set=test_set)
        # This function will save the task information into a json file named
        # `task_node_classification_1.json` and one numpy data file storing the
        # data splits, `train_set`, `val_set`, and `test_set`. The json file
        # will look like the following.

    .. code-block:: json

        {
            "description": "A node classification task for the example dataset.",
            "type": "NodeClassification",
            "feature": [
                "Node/DenseNodeFeature",
                "Node/SparseNodeFeature"
            ],
            "target": "Node/NodeLabel",
            "num_classes": 4,
            "train_set": {
                "file": "example_dataset__task_node_classification_1__4dcac617700f69a6dec06c2b5f75a246.npz",
                "key": "train_set"
            },
            "val_set": {
                "file": "example_dataset__task_node_classification_1__4dcac617700f69a6dec06c2b5f75a246.npz",
                "key": "val_set"
            },
            "test_set": {
                "file": "example_dataset__task_node_classification_1__4dcac617700f69a6dec06c2b5f75a246.npz",
                "key": "test_set"
            }
        }
    """  # noqa: E501,E262  #pylint: disable=line-too-long
    # Check the input arguments.
    assert isinstance(description, str), \
        "`description` must be a string."
    _check_feature(feature)
    _check_node_level_target(target)
    train_set, val_set, test_set = _check_data_splits(train_set, val_set,
                                                      test_set, train_ratio,
                                                      val_ratio, test_ratio,
                                                      num_samples)

    # Create the dictionary for task json file.
    task_dict = {
        "description": description,
        "type": "NodeClassification",
        "feature": feature,
        "target": target,
        "num_classes": num_classes
    }
    if train_set is not None:
        # Save the task data files, i.e., the data splits in this task.
        data_dict = {
            "train_set": np.array(train_set),
            "val_set": np.array(val_set),
            "test_set": np.array(test_set)
        }
        key_to_loc = save_data(f"{name}__task_node_classification_{task_id}",
                               save_dir=save_dir,
                               **data_dict)
        # Update the task dictionary with the data file names and keys.
        task_dict.update(key_to_loc)
        if isinstance(train_set[0], list):
            task_dict["num_splits"] = len(train_set)
    else:
        task_dict["train_ratio"] = train_ratio
        task_dict["val_ratio"] = val_ratio
        task_dict["test_ratio"] = test_ratio
        task_dict["num_samples"] = num_samples

    # Save the task json file.
    with open(os.path.join(save_dir,
                           f"task_node_classification_{task_id}.json"),
              "w",
              encoding="utf-8") as f:
        json.dump(task_dict, f, indent=4)

    return task_dict
