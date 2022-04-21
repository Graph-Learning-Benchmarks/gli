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
        return 1 + (max(map(_dict_depth, d.values()))
                    if d else 0)
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
    raise NotImplementedError