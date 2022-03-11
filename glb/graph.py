"""Base class of graph.

Any dataset needs to contain a graph instance. By default, a graph
instance should be initialized given by a metadata.json.
"""
import json
import os

import dgl
import torch
from dgl import DGLGraph

from .utils import load_data, is_sparse


def is_single_graph(data):
    """Return true if the glb data contains a single graph."""
    nodelist = data["Graph"]["_NodeList"]
    if nodelist.dim() == 1:
        return True
    elif nodelist.dim() == 2:
        return nodelist.shape[0] == 1
    else:
        raise ValueError("_NodeList has more than 2 dimensions.")


def get_single_graph(data, device="cpu"):
    """Initialize and return a single Graph instance given data."""
    edges = data["Edge"].pop("_Edge")  # (num_edges, 2)
    node_list = data["Graph"]["_NodeList"]
    src_nodes, dst_nodes = edges.T[0], edges.T[1]
    num_nodes = node_list.shape[-1]

    g: dgl.DGLGraph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes,
                                device=device)
    for attr, array in data["Node"].items():
        g.ndata[attr] = array

    for attr, array in data["Edge"].items():
        g.edges[attr] = array

    return g


def get_multi_graph(data, device="cpu"):
    """Initialize and return a list of Graph instance given data."""
    raise NotImplementedError


def read_glb_graph(metadata_path: os.PathLike, device="cpu", verbose=True) -> DGLGraph:
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
