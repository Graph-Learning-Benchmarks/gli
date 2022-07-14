"""Base class of graph.

Any dataset needs to contain a graph instance. By default, a graph
instance should be initialized given by a metadata.json.
"""
import json
import os
from copy import copy

import dgl
import scipy.sparse as sp
import torch

from .utils import file_reader, sparse_to_torch, sparse_to_dense_safe


class GLBGraph(dgl.DGLGraph):
    """Base class of GLB graph."""

    def node_to_dense(self, feat=None, node_group=None):
        """Convert node data to dense inplace.

        If both arguments are not provided, node_to_dense() will try to convert
        all node features to dense. (This only works for homograph.)

        Args:
            feat (str, optional): feature name. Defaults to None.
            node_group (str, optional): node group for heterograph. Defaults to
                None.

        Raises:
            NotImplementedError: If the graph is heterogeneous, feat and
                node_group cannot be None.
        """
        if self.is_homogeneous:
            if feat:
                self.ndata[feat] = sparse_to_dense_safe(self.ndata[feat])
            else:
                for k in self.ndata:
                    self.ndata[k] = sparse_to_dense_safe(self.ndata[k])
        else:
            if feat:
                assert node_group is not None
                self.ndata[feat][node_group] = sparse_to_dense_safe(
                    self.ndata[feat][node_group])
            else:
                raise NotImplementedError(
                    "Both feat and node_group should be provided for"
                    " heterograph."
                )

    def edge_to_dense(self, feat=None, edge_group=None):
        """Convert edge data to dense inplace.

        If both arguments are not provided, edge_to_dense() will try to convert
        all edge features to dense. (This only works for homograph.)

        Args:
            feat (str, optional): feature name. Defaults to None.
            edge_group (str, optional): edge group for heterograph. Defaults to
                None.

        Raises:
            NotImplementedError: If the graph is heterogeneous, feat and
                edge_group cannot be None.
        """
        if self.is_homogeneous:
            if feat:
                self.edata[feat] = sparse_to_dense_safe(self.edata[feat])
            else:
                for k in self.edata:
                    self.edata[k] = sparse_to_dense_safe(self.edata[k])
        else:
            if feat:
                assert edge_group is not None
                self.ndata[feat][edge_group] = sparse_to_dense_safe(
                    self.ndata[feat][edge_group])
            else:
                raise NotImplementedError(
                    "Both feat and edge_group should be provided for"
                    " heterograph."
                )

    def to_dense(self):
        """Convert both edge data and node data to dense inplace.

        This function only works for homograph.
        """
        if self.is_homogeneous:
            self.node_to_dense()
            self.edge_to_dense()
        else:
            raise NotImplementedError("to_dense only works for homograph.")


def read_glb_graph(metadata_path: os.PathLike, device="cpu", verbose=True):
    """Initialize and return a Graph instance given metadata.json."""
    pwd = os.path.dirname(metadata_path)
    with open(metadata_path, "r", encoding="utf-8") as fptr:
        metadata = json.load(fptr)

    if verbose:
        print(metadata["description"])

    assert _is_hetero_graph(metadata) == metadata[
        "is_heterogeneous"], "is_heterogeneous attribute is inconsistent"
    hetero = metadata["is_heterogeneous"]

    assert "data" in metadata, "attribute `data` not in metadata.json."

    for neg in ["Node", "Edge", "Graph"]:
        assert neg in metadata[
            "data"], f"attribute `{neg}` not in metadata.json"

    data = copy(metadata["data"])
    data = _dfs_read_file(pwd, data, device="cpu")

    if _is_single_graph(data):
        return _get_single_graph(data, device, hetero=hetero)
    else:
        return _get_multi_graph(data, device)


def _is_single_graph(data):
    """Return true if the glb data contains a single graph."""
    node_list = data["Graph"]["_NodeList"]
    if len(node_list.shape) == 1:
        return True
    elif len(node_list.shape) == 2:
        return node_list.shape[0] == 1
    else:
        raise ValueError("_NodeList has more than 2 dimensions.")


def _is_hetero_graph(data):
    """Return true if the glb data contains heterogeneous graph."""
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
        x = sparse_to_torch(x, to_dense=False, device=device)
    return x


def _get_single_graph(data, device="cpu", hetero=False):
    """Initialize and return a single Graph instance given data."""
    if hetero:
        g = _get_heterograph(data)
    else:
        g = _get_homograph(data)

    # REVIEW - Explicit class casting may lead to undefined behavior
    g.__class__ = GLBGraph

    return g.to(device=device)


def _get_multi_graph(data, device="cpu"):
    """Initialize and return a list of Graph instance given data."""
    node_list = data["Graph"].pop("_NodeList")
    edge_list = data["Graph"].pop("_EdgeList", None)
    graphs = []

    # Extract the whole graph
    g: dgl.DGLGraph = _get_single_graph(data)

    # Check array type before assigning
    for attr, array in data["Node"].items():
        g.ndata[attr] = _to_tensor(array)
    for attr, array in data["Edge"].items():
        g.edata[attr] = _to_tensor(array)

    # Decide subgraph types (node/edge-subgraph)
    subgraph_func = dgl.edge_subgraph if edge_list else dgl.node_subgraph
    entity_list = edge_list if edge_list else node_list

    # Transform indices into dense boolean tensor
    assert len(entity_list.shape) == 2, "NodeList/EdgeList should be a matrix."
    if sp.issparse(entity_list) and not sp.isspmatrix_csr(entity_list):
        # Only allow csr matrix
        entity_list = sp.csr_matrix(entity_list)
    for i in range(entity_list.shape[0]):
        if isinstance(entity_list, torch.Tensor):  # Dense pytorch tensor
            subgraph_entities = entity_list[i]
        elif isinstance(entity_list, sp.csr_matrix):
            subgraph_entities = entity_list.getrow(i).todense()
            subgraph_entities = torch.from_numpy(subgraph_entities).squeeze()
        subgraph_entities = subgraph_entities.bool()
        subgraph = subgraph_func(g, subgraph_entities).to(device=device)
        subgraph.__class__ = GLBGraph
        graphs.append(subgraph)

    for attr in data["Graph"]:
        for i, graph in enumerate(graphs):
            setattr(graph, attr, data["Graph"][attr])  # Graph-level features

    return graphs


def _get_homograph(data):
    """Get a homogeneous graph from data."""
    edges = data["Edge"].pop("_Edge")  # (num_edges, 2)
    src_nodes, dst_nodes = edges.T[0], edges.T[1]

    g: dgl.DGLGraph = dgl.graph((src_nodes, dst_nodes), device="cpu")

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


def _dict_depth(d):
    """Return the depth of a dictionary."""
    if isinstance(d, dict):
        return 1 + (max(map(_dict_depth, d.values())) if d else 0)
    return 0


def _dfs_read_file(pwd, d, device="cpu"):
    """Read file efficiently."""
    data = _dfs_read_file_helper(pwd, d, device)
    return data


def _dfs_read_file_helper(pwd, d, device="cpu"):
    """Read file recursively (helper of `_dfs_read_file`)."""
    if "file" in d:
        path = os.path.join(pwd, d["file"])
        array = file_reader.get(path, d.get("key"), device)
        return array

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
