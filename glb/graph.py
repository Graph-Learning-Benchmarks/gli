"""Base class of graph.

Any dataset needs to contain a graph instance. By default, a graph
instance should be initialized given by a metadata.json.
"""
import json
import os

import dgl
import torch

from .utils import load_data, is_sparse


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


def get_single_graph(data, device="cpu"):
    """Initialize and return a single Graph instance given data."""
    edges = data["Edge"].pop("_Edge")  # (num_edges, 2)
    node_list = data["Graph"]["_NodeList"]
    src_nodes, dst_nodes = edges.T[0], edges.T[1]

    g: dgl.DGLGraph = dgl.graph((src_nodes, dst_nodes), device=device)
    for attr, array in data["Node"].items():
        g.ndata[attr] = array

    for attr, array in data["Edge"].items():
        g.edges[attr] = array

    return g


def get_multi_graph(data, device="cpu"):
    """Initialize and return a list of Graph instance given data."""
    node_list = data["Graph"]["_NodeList"]
    edge_list = data["Graph"].get("_EdgeList", None)
    graphs = []

    # Extract the whole graph
    g: dgl.DGLGraph = get_single_graph(data, device)
    if edge_list:
        assert edge_list.dim() == 2, "_EdgeList should be a matrix."
        edge_list = edge_list > 0  # boolean tensor
        for i in len(edge_list):
            graphs.append(dgl.edge_subgraph(g, edge_list[i]))
    else:
        assert node_list.dim() == 2, "_NodeList should be a matrix."
        node_list = node_list > 0  # boolean tensor
        for i in len(edge_list):
            graphs.append(dgl.node_subgraph(g, node_list[i]))

    return graphs


def read_glb_graph(metadata_path: os.PathLike, device="cpu", verbose=True):
    """Initialize and return a Graph instance given metadata.json."""
    pwd = os.path.dirname(metadata_path)
    with open(metadata_path, "r", encoding="utf-8") as fptr:
        metadata = json.load(fptr)

    if verbose:
        print(metadata["description"])

    assert "data" in metadata, "attribute `data` not in metadata.json."

    for neg in ["Node", "Edge", "Graph"]:
        assert neg in metadata[
            "data"], f"attribute `{neg}` not in metadata.json"

    assert "_Edge" in metadata["data"]["Edge"]
    assert "_NodeList" in metadata["data"]["Graph"]

    if not is_hetero_graph(metadata):
        raise NotImplementedError("Does not support Heterogeneous graph yet.")

    data_buffer = {}
    data = {"Node": {}, "Edge": {}, "Graph": {}}
    for neg in ["Node", "Edge", "Graph"]:
        for attr, props in metadata["data"][neg].items():
            if "file" in props:
                filename = props["file"]
                if filename not in data_buffer:
                    raw = load_data(os.path.join(pwd, filename))
                    data_buffer[filename] = raw
                else:
                    raw = data_buffer[filename]
                if "key" in props:
                    key = props["key"]
                    array = raw[key]
                else:
                    array = raw
                if is_sparse(array):
                    array = array.all().toarray()
                array = torch.from_numpy(array).to(device=device)
                data[neg][attr] = array

    if is_single_graph(data):
        return get_single_graph(data, device)
    else:
        return get_multi_graph(data, device)


def _dict_depth(d):
    """Return the depth of a dictionary."""
    if isinstance(d, dict):
        return 1 + (max(map(_dict_depth, d.values()))
                    if d else 0)
    return 0
