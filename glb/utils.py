"""Utility functions."""
import os

import numpy as np
import scipy.sparse as sp
import torch


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


def is_sparse(array):
    """Return true if array is sparse.

    This method is to deal with the situation where array is loaded from
    sparse matrix by np.load(), which will wrap array to be a numpy.ndarray.
    """
    if sp.issparse(array):
        return True
    if isinstance(array, np.ndarray):
        return array.dtype.kind not in set("buifc")
    raise TypeError


class KeyedFileReader():
    def __init__(self) -> None:
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
            array = raw[key]
        else:
            array = raw
        
        if is_sparse(array):
            array = array.all().toarray()
        return torch.from_numpy(array).to(device=device)

file_reader = KeyedFileReader()