""":mod:`gli.graph` graph loading module.

The :mod:`gli.graph` module implements the loading of a graph from local files
in the GLI data format.
"""
import json
import os
from copy import copy

import dgl
import scipy.sparse as sp
import torch
import numpy as np
from tqdm import tqdm

from .utils import sparse_to_torch, load_data


def read_gli_graph(metadata_path: os.PathLike, device="cpu", verbose=True):
    """Read a local `metadata.json` file and return a (or a list of) graph(s).

    :func:`gli.graph.read_gli_graph` reads a graph or a list of graphs
    according to the `metadata.json` file.

    :param metadata_path: path to the `metadata.json` file.
    :type metadata_path: :class:`os.PathLike`
    :param device: device name, defaults to “cpu”.
    :type device: str, optional
    :param verbose: verbose level, defaults to False.
    :type verbose: bool, optional

    :rtype: :class:`dgl.DGLGraph` or a list of :class:`dgl.DGLGraph`

    .. important::
        The file format of a `metadata.json` is documented in :ref:`format`.

    Notes
    -----
    This function is used to read a GLI task file locally. It is not used to
    fetch a task configuration from a remote server. If you want to download
    any task configuration provided by GLI, use
    :func:`gli.dataloading.get_gli_task` instead.

    Additionally, this function is useful when you want to test loading a new
    task configuration file locally.

    """
    pwd = os.path.dirname(metadata_path)
    with open(metadata_path, "r", encoding="utf-8") as fptr:
        metadata = json.load(fptr)

    if verbose:
        print(metadata["description"])

    assert _is_hetero_graph(metadata) == metadata[
        "is_heterogeneous"], "is_heterogeneous attribute is inconsistent"
    hetero = metadata["is_heterogeneous"]
    name = metadata["description"]

    assert "data" in metadata, "attribute `data` not in metadata.json."

    for neg in ["Node", "Edge", "Graph"]:
        assert neg in metadata[
            "data"], f"attribute `{neg}` not in metadata.json"

    data = copy(metadata["data"])
    data = _dfs_read_file(pwd, data, device="cpu")

    if _is_single_graph(data):
        return _get_single_graph(data, device, hetero=hetero, name=name)
    else:
        return _get_multi_graph(data, device, name=name)


def _is_single_graph(data):
    """Return true if the gli data contains a single graph."""
    node_list = data["Graph"]["_NodeList"]
    if len(node_list.shape) == 1:
        return True
    elif len(node_list.shape) == 2:
        return node_list.shape[0] == 1
    else:
        raise ValueError("_NodeList has more than 2 dimensions.")


def _is_hetero_graph(data):
    """Return true if the gli data contains heterogeneous graph."""
    depth = _dict_depth(data)
    # Heterogeneous graph has one more depth than a homogeneous one.
    if depth == 5:
        return True
    elif depth == 4:
        return False
    else:
        raise RuntimeError("metadata.json has wrong structure.")


def _to_tensor(x, device="cpu"):
    """Wrap x into a tensor."""
    if not torch.is_tensor(x) and sp.issparse(x):
        x = sparse_to_torch(x, convert_to_dense=False, device=device)
    return x


def _get_single_graph(data, device="cpu", hetero=False, name=None):
    """Initialize and return a single Graph instance given data."""
    if hetero:
        g = _get_heterograph(data)
    else:
        g = _get_homograph(data)

    setattr(g, "name", name)
    return g.to(device=device)


def _get_multi_graph(data, device="cpu", name=None):
    """Initialize and return a list of Graph instance given data."""
    # Extract the whole graph
    g: dgl.DGLGraph = _get_single_graph(data)

    node_list = data["Graph"].pop("_NodeList")
    edge_list = data["Graph"].pop("_EdgeList", None)
    graphs = []

    # Check array type before assigning
    for attr, array in data["Node"].items():
        g.ndata[attr] = _to_tensor(array)
    for attr, array in data["Edge"].items():
        g.edata[attr] = _to_tensor(array)

    if edge_list is None:
        # Infer edge_list from node_list
        edges = torch.stack(g.edges()).T
        graph_edge_matrix = _get_graph_edge_matrix(node_list, edges)
        edge_list = graph_edge_matrix >= 2

    edge_list = edge_list.tolil()

    for i in tqdm(range(edge_list.shape[0]), desc="Processing graphs"):
        if isinstance(edge_list, torch.Tensor):
            subgraph_edges = edge_list[i]
            subgraph_edges = subgraph_edges.bool()
        elif isinstance(edge_list, sp.lil_matrix):
            subgraph_edges = edge_list.rows[i]
        subgraph = dgl.edge_subgraph(g, subgraph_edges).to(device)
        setattr(subgraph, "name", name)
        graphs.append(subgraph)

    for attr in data["Graph"]:
        for i, graph in enumerate(graphs):
            # Graph-level features
            setattr(graph, attr, data["Graph"][attr][i])

    return graphs


def _get_homograph(data):
    """Get a homogeneous graph from data."""
    edges = data["Edge"].pop("_Edge")  # (num_edges, 2)
    src_nodes, dst_nodes = edges.T[0], edges.T[1]
    num_nodes = data["Graph"]["_NodeList"].shape[-1]
    assert edges.max() < num_nodes, ("The largest node id exceeds num_nodes."
                                     "Is the node id zero-based indexed?")

    g: dgl.DGLGraph = dgl.graph((src_nodes, dst_nodes),
                                num_nodes=num_nodes,
                                device="cpu")

    for attr, array in data["Node"].items():
        g.ndata[attr] = _to_tensor(array)

    for attr, array in data["Edge"].items():
        g.edata[attr] = _to_tensor(array)
    return g


def _get_heterograph(data):
    """Get heterogeneous graph."""
    node_depth = _dict_depth(data["Node"])
    node_classes = []
    node_features = {}
    edge_features = {}
    num_nodes_dict = {}
    num_nodes = data["Graph"]["_NodeList"].shape[-1]
    node_to_class = torch.zeros(num_nodes, dtype=torch.int)
    node_map = torch.zeros(num_nodes, dtype=torch.int)
    if node_depth == 1:
        # Nodes are homogeneous
        node_classes.append("Node")
        node_features["Node"] = data["Node"]
        # node_features["Node"].pop("_ID", None)
        num_nodes_dict["Node"] = num_nodes
        node_map = torch.arange(num_nodes, dtype=torch.int)
    else:
        for i, node_class in enumerate(data["Node"]):
            node_classes.append(node_class)
            idx = data["Node"][node_class]["_ID"]
            node_to_class[idx] = i
            node_map[idx] = torch.arange(len(idx), dtype=torch.int)
            node_features[node_class] = data["Node"][node_class]
            # node_features[node_class].pop("_ID", None)
            num_nodes_dict[node_class] = len(idx)

    edge_depth = _dict_depth(data["Edge"])
    assert edge_depth == 2, "Edges of heterograph must be heterogeneous, too."

    graph_data = {}  # triplet to (src_tensor, dst_tensor)
    for edge_class in data["Edge"]:
        # Infer triplet
        edges = data["Edge"][edge_class]["_Edge"]
        src_class = node_classes[node_to_class[edges[0][0]]]
        dst_class = node_classes[node_to_class[edges[0][1]]]
        triplet = (src_class, edge_class, dst_class)
        edges = node_map[edges]
        graph_data[triplet] = (edges.T[0], edges.T[1])
        edge_features[edge_class] = data["Edge"][edge_class]
        # edge_features[edge_class].pop("_ID", None)

    g: dgl.DGLGraph = dgl.heterograph(graph_data,
                                      num_nodes_dict=num_nodes_dict)

    # Set indexing map
    setattr(g, "node_classes", node_classes)
    setattr(g, "node_to_class", node_to_class)
    setattr(g, "node_map", node_map)

    # Add node and edge features
    for node_class, node_feats in node_features.items():
        for feat_name, feat_tensor in node_feats.items():
            if len(g.ntypes) == 1:
                g.ndata[feat_name] = _to_tensor(feat_tensor)
            else:
                g.ndata[feat_name] = {node_class: _to_tensor(feat_tensor)}
    for edge_class, edge_feats in edge_features.items():
        for feat_name, feat_tensor in edge_feats.items():
            g.edata[feat_name] = {edge_class: _to_tensor(feat_tensor)}

    return g


def _dict_depth(d):
    """Return the depth of a dictionary."""
    if isinstance(d, dict):
        return 1 + (max(map(_dict_depth, d.values())) if d else 0)
    return 0


def _dfs_read_file(pwd, d, device="cpu"):
    """Read file efficiently."""
    return _dfs_read_file_helper(pwd, d, device)


def _dfs_read_file_helper(pwd, d, device="cpu"):
    """Read file recursively (helper of `_dfs_read_file`)."""
    if "file" in d:
        path = os.path.join(pwd, d["file"])
        return load_data(path, d.get("key"), device)

    empty_keys = []
    for k in d:
        entry = _dfs_read_file_helper(pwd, d[k], device=device)
        if entry is None:
            empty_keys.append(k)
        else:
            d[k] = entry
    for k in empty_keys:
        d.pop(k)
    return d


def _get_graph_edge_matrix(graph_node_matrix: sp.spmatrix,
                           edges: torch.Tensor):
    """Get graph edge csr matrix."""
    if isinstance(graph_node_matrix, torch.Tensor):
        graph_node_matrix = graph_node_matrix.numpy()
    graph_node_matrix = graph_node_matrix.astype(np.int8)
    n_nodes = graph_node_matrix.shape[1]
    n_edges = edges.shape[0]
    edge_id = torch.arange(0, edges.shape[0])  # (n_edge,)
    indices = torch.stack(
        (edges, edge_id.repeat(2, 1).T),
        dim=2)  # (2 [nodes in a edge], n_edge, 2 [repeat edge_id])
    i = torch.cat((indices[:, 0, :], indices[:, 1, :]), dim=0)
    data = np.ones(i.shape[0])
    node_edge_matrix = sp.coo_matrix((data, i.T),
                                     shape=(n_nodes, n_edges),
                                     dtype=np.int8)
    graph_edge_matrix = graph_node_matrix @ node_edge_matrix
    return graph_edge_matrix
