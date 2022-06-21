"""Utility functions."""
import json
import os

import numpy as np
import scipy.sparse as sp
import torch
import dgl


def load_data(path: os.PathLike):
    """Load data from given path.

    Supported file format:
    1. .npz
    2. .npy
    """
    _, ext = os.path.splitext(path)
    if ext in (".npz", ".npy"):
        data = np.load(path, allow_pickle=True)
    else:
        raise NotImplementedError(f"{ext} file is currently not supported.")
    return data


def unwrap_array(array):
    """Unwrap the array.

    This method is to deal with the situation where array is loaded from
    sparse matrix by np.load(), which will wrap array to be a numpy.ndarray.
    """
    if isinstance(array, np.ndarray):
        if array.dtype.kind not in set("buifc"):
            return array.all()
    return array


class KeyedFileReader():
    """File reader for npz files."""

    def __init__(self) -> None:
        """File reader for npz files."""
        self._data_buffer = {}

    def get(self, path, key=None, device="cpu"):
        """Return a torch array.

        TODO:
            - Check sparsity and return sparse torch array if needed.
        """
        if path not in self._data_buffer:
            raw = load_data(path)
            self._data_buffer[path] = raw
        else:
            raw = self._data_buffer[path]

        if key:
            array = raw.get(key, None)
        else:
            array = raw

        if array is None:
            return None

        assert isinstance(array, np.ndarray)

        array = unwrap_array(array)

        if sp.issparse(array):
            # Keep the array format to be scipy rather than pytorch
            return array
        else:
            return torch.from_numpy(array).to(device=device)


file_reader = KeyedFileReader()


def sparse_to_torch(sparse_array: sp.spmatrix, to_dense=True, device="cpu"):
    """Transform a sparse scipy array to sparse(coo) torch tensor.

    Note - add csr support.
    """
    if to_dense:
        array = sparse_array.toarray()
        return torch.from_numpy(array).to(device)

    else:
        sparse_array = sp.coo_matrix(sparse_array)
        i = torch.LongTensor(np.vstack((sparse_array.row, sparse_array.col)))
        v = torch.FloatTensor(sparse_array.data)
        shape = sparse_array.shape

        return torch.sparse_coo_tensor(i, v, torch.Size(shape),
                                       device=device).to_sparse_csr()


def dgl_to_glb(graph: dgl.DGLGraph,
               name: str,
               pdir: os.PathLike = None,
               **kwargs):
    """Dump a dgl graph into glb format."""
    metadata = {"data": {"Node": {}, "Edge": {}, "Graph": {}}}
    metadata.update(kwargs)
    npz = f"{name}.npz"
    data = {}
    if graph.is_multigraph:
        raise NotImplementedError
    if graph.is_homogeneous:
        for k, v in graph.ndata.items():
            entry = f"node_{k}"
            data[entry] = v.cpu().numpy()
            metadata["data"]["Node"][entry] = {"file": npz, "key": entry}
        for k, v in graph.edata.items():
            entry = f"edge_{k}"
            data[entry] = v.cpu().numpy()
            metadata["data"]["Edge"][entry] = {"file": npz, "key": entry}

        # Reserved Entries
        entry = "_Edge"
        data[entry] = torch.stack(graph.edges()).T.cpu().numpy()
        metadata["data"]["Edge"]["_Edge"] = {"file": npz, "key": entry}

    else:

        for node_type in graph.ntypes:
            # Save node id
            entry = f"node_{node_type}_id"
            metadata["data"]["Node"][node_type] = {
                "_ID": {
                    "file": npz,
                    "key": entry
                }
            }
            data[entry] = graph.nodes(node_type).cpu().numpy()
            # Save node features
            for k, v in graph.ndata.items():
                if node_type in v:
                    entry = f"node_{node_type}_{k}"
                    data[entry] = v.cpu().numpy()
                    metadata["data"]["Node"][node_type][entry] = {
                        "file": npz,
                        "key": entry
                    }

        edge_id = 0
        for edge_type in graph.etypes:
            # Save edge id
            entry = f"edge_{edge_type}_id"
            metadata["data"]["Edge"][edge_type] = {
                "_ID": {
                    "file": npz,
                    "key": entry
                },
            }
            u, v = graph.edges(etype=edge_type)
            data[entry] = graph.edge_ids(u, v, edge_type) + edge_id
            edge_id += len(u)
            # Save edges
            entry = f"edge_{edge_type}"
            metadata["data"]["Edge"][edge_type]["_Edge"] = {
                "file": npz,
                "key": entry
            }
            data[entry] = torch.stack(graph.edges(edge_type)).T.cpu().numpy()
            # Save edge features
            for k, v in graph.edata.items():
                # FIXME - AssertionError: Current HeteroNodeDataView
                # has multiple node types, can not be iterated.
                if edge_type in v:
                    entry = f"edge_{edge_type}_{k}"
                    data[entry] = v.cpu().numpy()
                    metadata["data"]["Edge"][entry] = {
                        "file": npz,
                        "key": entry
                    }

    entry = "_NodeList"
    data[entry] = np.ones((1, graph.num_nodes()))
    metadata["data"]["Graph"]["_NodeList"] = {"file": npz, "key": entry}

    entry = "_EdgeList"
    data[entry] = np.ones((1, graph.num_edges()))
    metadata["data"]["Graph"]["_EdgeList"] = {"file": npz, "key": entry}

    # Save file
    os.makedirs(pdir)
    npz_path = os.path.join(pdir, npz)
    metadata_path = os.path.join(pdir, "metadata.json")

    np.savez_compressed(npz_path, **data)
    with open(metadata_path, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp)

    raise NotImplementedError
