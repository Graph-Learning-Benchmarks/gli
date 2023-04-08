"""Helper functions for creating datasets in GLI format."""
import json
import os
import warnings
import numpy as np
from scipy.sparse import isspmatrix

from gli.utils import save_data


def detect_array_type(array):
    """Detect the type of the data in the array."""
    if isspmatrix(array) or isinstance(array, np.ndarray):
        if array.size == 0:
            raise ValueError("The input array is empty.")

        # Check for the first non-null element's type
        for element in array.data:
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

        Args:
            name (str): The name of the attribute.
            data (array-like): The data of the attribute.
            description (str, optional): The description of the attribute.
            data_type (str, optional): The type of the data. If not specified,
                the type will be automatically detected.
            data_format (str, optional): The format of the data. If not
                specified, the format will be automatically detected.
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
    Save the graph information to metadata.json and Numpy data files.

    Args:
        name (str): The name of the graph dataset.
        edge (array): An array of shape (num_edges, 2). Each row is an edge
            between the two nodes with the given node IDs.
        node_attrs (list of Attribute, optional): A list of attributes of the
            nodes.
        edge_attrs (list of Attribute, optional): A list of attributes of the
            edges.
        graph_node_lists ((sparse) array, optional): An array of shape
            (num_graphs, num_nodes). Each row corresponds to a graph and each
            column corresponds to a node. The value of the element (i, j) is 1
            if node j is in graph i, otherwise 0. If not specified, the graph
            will be considered as a single graph.
        graph_edge_lists ((sparse) array, optional): An array of shape
            (num_graphs, num_edges). Each row corresponds to a graph and each
            column corresponds to an edge. The value of the element (i, j) is 1
            if edge j is in graph i, otherwise 0. If not specified, the edges
            contained in each graph specified by `graph_node_lists` will be
            considered as all the edges among the nodes in the graph.
        graph_attrs (list of Attribute, optional): A list of attributes of the
            graphs.
        description (str, optional): The description of the dataset.
        citation (str, optional): The citation of the dataset.
        is_heterogeneous (bool, optional): Whether the graph is heterogeneous.
        save_dir (str, optional): The directory to save the Numpy data files
            and metadata.json.
    """
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
                    raise ValueError("The length of data of all attributes of"
                                     f"the {object_name} must be the same.")

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
