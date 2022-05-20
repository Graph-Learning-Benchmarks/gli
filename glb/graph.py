"""Base class of graph.

Any dataset needs to contain a graph instance. By default, a graph
instance should be initialized given by a metadata.json.
"""
from copy import copy
import json
import os

import dgl
import torch

from .utils import file_reader


def is_single_graph(data):
    """Return true if the glb data contains a single graph."""
    nodelist = data["Graph"]["_NodeList"]
    nodelist = torch.squeeze(nodelist)
    if nodelist.dim() == 1:
        return True
    elif nodelist.dim() == 2:
        return nodelist.shape[0] == 1
    else:
        raise ValueError("_NodeList has more than 2 dimensions.")


def is_hetero_graph(data):
    """Return true if the glb data contains heterogeneous graph."""
    depth = _dict_depth(data)
    # Heterogeneous graph has one more depth than a homogeneous one.
    if depth == 5:
        return True
    elif depth == 4:
        return False
    else:
        raise RuntimeError("metadata.json has wrong structure.")


def get_single_graph(data, device="cpu", hetero=False):
    """Initialize and return a single Graph instance given data."""
    if hetero:
        g = get_heterograph(data, device=device)
    else:
        edges = data["Edge"].pop("_Edge")  # (num_edges, 2)
        src_nodes, dst_nodes = edges.T[0], edges.T[1]

        g: dgl.DGLGraph = dgl.graph((src_nodes, dst_nodes), device=device)
        for attr, array in data["Node"].items():
            g.ndata[attr] = array

        for attr, array in data["Edge"].items():
            g.edata[attr] = array

    return g


def get_multi_graph(data, device="cpu"):
    """Initialize and return a list of Graph instance given data."""
    node_list = data["Graph"].pop("_NodeList")
    edge_list = data["Graph"].pop("_EdgeList", None)
    graphs = []

    # Extract the whole graph
    g: dgl.DGLGraph = get_single_graph(data, device)
    for attr, array in data["Node"].items():
        g.ndata[attr] = array
    for attr, array in data["Edge"].items():
        g.edata[attr] = array

    if edge_list:
        edge_list = edge_list.bool()
        assert edge_list.dim() == 2, "_EdgeList should be a matrix."
        edge_list = edge_list > 0
        for i in range(len(edge_list)):
            graphs.append(dgl.edge_subgraph(g, edge_list[i]))
    else:
        node_list = node_list.bool()
        assert node_list.dim() == 2, "_NodeList should be a matrix."
        for i in range(len(node_list)):
            graphs.append(dgl.node_subgraph(g, node_list[i]))

    for attr in data["Graph"]:
        for i, graph in enumerate(graphs):
            setattr(graph, attr, data["Graph"][attr])  # Graph-level features

    return graphs


def read_glb_graph(metadata_path: os.PathLike, device="cpu", verbose=True):
    """Initialize and return a Graph instance given metadata.json."""
    pwd = os.path.dirname(metadata_path)
    with open(metadata_path, "r", encoding="utf-8") as fptr:
        metadata = json.load(fptr)

    if verbose:
        print(metadata["description"])

    hetero = is_hetero_graph(metadata)

    assert "data" in metadata, "attribute `data` not in metadata.json."

    for neg in ["Node", "Edge", "Graph"]:
        assert neg in metadata[
            "data"], f"attribute `{neg}` not in metadata.json"

    data = copy(metadata["data"])
    data = dfs_read_file(pwd, data, device=device)

    if is_single_graph(data):
        return get_single_graph(data, device, hetero=hetero)
    else:
        return get_multi_graph(data, device)


def _dict_depth(d):
    """Return the depth of a dictionary."""
    if isinstance(d, dict):
        return 1 + (max(map(_dict_depth, d.values())) if d else 0)
    return 0


def dfs_read_file(pwd, d, device="cpu"):
    if "file" in d:
        path = os.path.join(pwd, d["file"])
        array = file_reader.get(path, d.get("key"), device)
        return array
    else:
        for k in d:
            d[k] = dfs_read_file(pwd, d[k], device=device)
        return d


def get_heterograph(data, device="cpu"):
    node_depth = _dict_depth(data["Node"])
    node_classes = []
    node_features = {}
    edge_features = {}
    num_nodes_dict = {}
    num_nodes = data["Graph"]["_NodeList"].shape[-1]
    node_to_class = torch.zeros(num_nodes, dtype=torch.int)
    if node_depth == 1:
        # Nodes are homogeneous
        node_classes.append("Node")
        node_features["Node"] = data["Node"]
        node_features["Node"].pop("_ID", None)
        num_nodes_dict["Node"] = num_nodes
    else:
        for i, node_class in enumerate(data["Node"]):
            node_classes.append(node_class)
            idx = data["Node"][node_class]["_ID"]
            node_to_class[idx] = i
            node_features[node_class] = data["Node"][node_class]
            node_features[node_class].pop("_ID", None)
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
        graph_data[triplet] = (edges.T[0], edges.T[1])
        edge_features[edge_class] = data["Edge"][edge_class]
        edge_features[edge_class].pop("_ID", None)

    g: dgl.DGLGraph = dgl.heterograph(graph_data,
                                      num_nodes_dict=num_nodes_dict)

    # Add node and edge features
    for node_class, node_feats in node_features.items():
        for feat_name, feat_tensor in node_feats.items():
            if len(g.ntypes) == 1:
                g.ndata[feat_name] = feat_tensor
            else:
                g.ndata[feat_name] = {node_class: feat_tensor}
    for edge_class, edge_feats in edge_features.items():
        for feat_name, feat_tensor in edge_feats.items():
            g.edata[feat_name] = {edge_class: feat_tensor}

    return g
