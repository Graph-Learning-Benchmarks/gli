"""Utility functions."""
import os
import numpy as np
import scipy.sparse as sp


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
