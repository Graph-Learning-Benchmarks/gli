"""Utility functions.

Download functions for Google Drive come from
https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py.
"""
import contextlib
import itertools
import json
import os
import re
import subprocess
import warnings

from typing import (Iterator, Optional, Tuple)
from urllib.parse import urlparse

import dgl
import numpy as np
import requests
import scipy.sparse as sp
import torch
from torch.utils.model_zoo import tqdm

from gli import ROOT_PATH, WARNING_DENSE_SIZE


def _save_response_content(
    content: Iterator[bytes],
    destination: str,
    length: Optional[int] = None,
    verbose: Optional[bool] = False,
) -> None:
    with open(destination, "wb") as fh, tqdm(total=length,
                                             disable=not verbose) as pbar:
        for chunk in content:
            # filter out keep-alive new chunks
            if not chunk:
                continue

            fh.write(chunk)
            pbar.update(len(chunk))


def _get_google_drive_file_id(url: str) -> Optional[str]:
    parts = urlparse(url)

    if re.match(r"(drive|docs)[.]google[.]com", parts.netloc) is None:
        return None

    match = re.match(r"/file/d/(?P<id>[^/]*)", parts.path)
    if match is None:
        match = re.match(r"id=(?P<id>[^/]*)&export=download", parts.query)
        if match is None:
            return None
        return match.group("id")

    return match.group("id")


def _extract_gdrive_api_response(
        response,
        chunk_size: int = 32 * 1024) -> Tuple[bytes, Iterator[bytes]]:
    content = response.iter_content(chunk_size)
    first_chunk = None
    # filter out keep-alive new chunks
    while not first_chunk:
        first_chunk = next(content)
    content = itertools.chain([first_chunk], content)

    try:
        match = re.search(
            "<title>Google Drive - (?P<api_response>.+?)</title>",
            first_chunk.decode())  # noqa
        api_response = match["api_response"] if match is not None else None
    except UnicodeDecodeError:
        api_response = None
    return api_response, content


def download_file_from_google_drive(g_url: str,
                                    root: str,
                                    filename: Optional[str] = None,
                                    verbose: Optional[bool] = False):
    """Download a Google Drive file from  and place it in root.

    Args:
        g_url (str): Google Drive url of file to be downloaded
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the
            id of the file.
    """
    # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url  # noqa pylint: disable=line-too-long

    file_id = _get_google_drive_file_id(g_url)
    root = os.path.expanduser(root)
    if not filename:
        filename = file_id
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    url = "https://drive.google.com/uc"
    params = dict(id=file_id, export="download")
    with requests.Session() as session:
        response = session.get(url, params=params, stream=True)

        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                token = value
                break
        else:
            api_response, content = _extract_gdrive_api_response(response)
            token = "t" if api_response == "Virus scan warning" else None

        if token is not None:
            response = session.get(url,
                                   params=dict(params, confirm=token),
                                   stream=True)
            api_response, content = _extract_gdrive_api_response(response)

        if api_response == "Quota exceeded":
            raise RuntimeError(
                f"The daily quota of the file {filename} is exceeded and it "
                f"can't be downloaded. This is a limitation of Google Drive "
                f"and can only be overcome by trying again later.")

        _save_response_content(content, fpath, verbose=verbose)

    # In case we deal with an unhandled GDrive API response, the file should be smaller than 10kB and contain only text  # noqa pylint: disable=line-too-long
    if os.stat(fpath).st_size < 10 * 1024:
        with contextlib.suppress(UnicodeDecodeError), open(fpath) as fh:  # noqa pylint: disable=unspecified-encoding, line-too-long
            text = fh.read()
            # Regular expression to detect HTML. Copied from https://stackoverflow.com/a/70585604  # noqa pylint: disable=line-too-long
            if re.search(
                    r"</?\s*[a-z-][^>]*\s*>|(&(?:[\w\d]+|#\d+|#x[a-f\d]+);)",
                    text):  # noqa
                warnings.warn(
                    f"We detected some HTML elements in the downloaded file. "
                    f"This most likely means that the download triggered an unhandled API response by GDrive. "  # noqa pylint: disable=line-too-long
                    f"Please report this to torchvision at https://github.com/pytorch/vision/issues including "  # noqa pylint: disable=line-too-long
                    f"the response:\n\n{text}")
    elif verbose:
        print(f"Successfully downloaded {filename} to {root} from {g_url}.")


def load_data(path, key=None, device="cpu"):
    """Load data from npy or npz file, return sparse array or torch tensor.

    Parameters
    ----------
    path : str
        Path to data file
    key : str, optional
        by default None
    device : str, optional
        by default "cpu"

    Returns
    -------
    torch.Tensor or scipy.sparse.matrix

    Raises
    ------
    TypeError
        Unrecognized file extension
    """
    _, ext = os.path.splitext(path)
    if ext not in (".npz", ".npy"):
        raise TypeError(f"Invalid file extension {ext}.")

    if path.endswith(".sparse.npz"):
        # Sparse matrix
        assert key is None, "Sparse format cannot contain key."
        return sp.load_npz(path)

    # Dense arrays file with a key
    raw = np.load(path, allow_pickle=False)
    assert key is not None
    array: np.ndarray = raw.get(key)
    raw.close()
    try:
        torch_tensor = torch.from_numpy(array)
    except TypeError:
        print(f"Non supported np.ndarray type {array.dtype}.")
        return None
    return torch_tensor.to(device)


def sparse_to_torch(sparse_array: sp.spmatrix,
                    convert_to_dense=False,
                    device="cpu"):
    """Transform a sparse scipy array to sparse(coo) torch tensor."""
    if convert_to_dense:
        array = sparse_array.toarray()
        return torch.from_numpy(array).to(device)

    else:
        sparse_type = sparse_array.getformat()
        shape = sparse_array.shape
        if sparse_type == "coo":
            i = torch.LongTensor(
                np.vstack((sparse_array.row, sparse_array.col)))
            v = torch.FloatTensor(sparse_array.data)

            return torch.sparse_coo_tensor(i,
                                           v,
                                           torch.Size(shape),
                                           device=device)

        elif sparse_type == "csr":
            sparse_array: sp.csr_matrix
            crow_indices = sparse_array.indptr
            col_indices = sparse_array.indices
            values = sparse_array.data
            return torch.sparse_csr_tensor(crow_indices,
                                           col_indices,
                                           values,
                                           size=torch.Size(shape),
                                           device=device)

        else:
            raise TypeError(f"Unsupported sparse type {sparse_type}")


def download_data(dataset: str, verbose=False):
    """Download dependent data of a configuration (metadata/task) file.

    Args:
        dataset (str): Name of dataset.
        filename (str): Name of configuration file. e.g., `metadata.json`.
        verbose (bool, optional): Defaults to False.
    """
    data_dir = os.path.join(ROOT_PATH, "datasets/", dataset)
    if os.path.isdir(data_dir):
        url_file = os.path.join(data_dir, "urls.json")
    else:
        raise FileNotFoundError(f"cannot find dataset {dataset}.")
    if not os.path.exists(url_file):
        raise FileNotFoundError(f"cannot find url files of {dataset}.")
    with open(url_file, "r", encoding="utf-8") as fp:
        url_dict = json.load(fp)
    for data_file_name, url in url_dict.items():
        data_file_path = os.path.join(data_dir, data_file_name)
        if os.path.exists(data_file_path):
            if verbose:
                print(f"{data_file_path} already exists. Skip downloading.")
            continue
        else:
            _download(url, data_file_path, verbose=verbose)


def _download(url, out, verbose=False):
    """Download url to out by running a wget subprocess or a gdrive downloader.

    Note - This function may generate a lot of unhelpful message.
    """
    parts = urlparse(url)
    if re.match(r"(drive|docs)[.]google[.]com", parts.netloc) is not None:
        root, filename = os.path.split(out)
        download_file_from_google_drive(url, root, filename, verbose)
        return

    if verbose:
        subprocess.run(["wget", "-O", out, url], check=True)
    else:
        subprocess.run(["wget", "-q", "-O", out, url], check=True)


def _sparse_to_dense_safe(array: torch.Tensor):
    """Convert a sparse tensor to dense.

    Throw user warning if the dense array size is larger than 1 G.

    Args:
        array (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Dense tensor.
    """
    if array.is_sparse or array.is_sparse_csr:
        array = array.to_dense()
        array_size = array.element_size() * array.nelement()
        if array_size > WARNING_DENSE_SIZE:
            warnings.warn(
                f"Trying to convert a large sparse tensor to a dense tensor. "
                f"The dense tensor occupies {array_size} bytes.")
    return array


def _to_dense(graph: dgl.DGLGraph, feat=None, group=None, is_node=True):
    graph_data = graph.ndata if is_node else graph.edata

    if graph.is_homogeneous:
        if feat:
            graph_data[feat] = _sparse_to_dense_safe(graph_data[feat])
        else:
            for k in graph_data:
                graph_data[k] = _sparse_to_dense_safe(graph_data[k])
    elif feat:
        assert group is not None
        graph.ndata[feat][group] = _sparse_to_dense_safe(
            graph.ndata[feat][group])
    else:
        raise NotImplementedError("Both feat and group should be provided for"
                                  " heterograph.")

    return graph


def edge_to_dense(graph: dgl.DGLGraph, feat=None, edge_group=None):
    """Convert edge data to dense.

    If both arguments are not provided, edge_to_dense() will try to convert
    all edge features to dense. (This only works for homograph.)

    Args:
        graph (dgl.DGLGraph): graph whose edges will be converted to dense.
        feat (str, optional): feature name. Defaults to None.
        edge_group (str, optional): edge group for heterograph. Defaults to
            None.

    Raises:
        NotImplementedError: If the graph is heterogeneous, feat and
            edge_group cannot be None.
    """
    return _to_dense(graph, feat, edge_group, is_node=False)


def node_to_dense(graph: dgl.DGLGraph, feat=None, node_group=None):
    """Convert node data to dense.

    If both arguments are not provided, node_to_dense() will try to convert
    all node features to dense. (This only works for homograph.)

    Args:
        graph (dgl.DGLGraph): graph whose nodes will be converted to dense.
        feat (str, optional): feature name. Defaults to None.
        node_group (str, optional): node group for heterograph. Defaults to
            None.

    Raises:
        NotImplementedError: If the graph is heterogeneous, feat and
            node_group cannot be None.
    """
    return _to_dense(graph, feat, node_group, is_node=True)


def to_dense(graph: dgl.DGLGraph):
    """Convert data to dense.

    This function only works for homograph.

    Args:
        graph (dgl.DGLGraph): graph whose data will be converted to dense.

    Raises:
        NotImplementedError: If the graph is heterogeneous.
    """
    if graph.is_homogeneous:
        return node_to_dense(edge_to_dense(graph))
    else:
        raise NotImplementedError("to_dense only works for homograph.")


def save_data(prefix, **kwargs):
    """Save arrays into numpy binary formats.

    Dense arrays (numpy) will be saved in the below format as a single file:
        <prefix>.npz
    Sparse arrays (scipy) will be saved in the below format individually:
        <prefix>_<key>.sparse.npz
    """
    dense_arrays = {}
    sparse_arrays = {}
    for key, matrix in kwargs.items():
        if sp.issparse(matrix):
            sparse_arrays[key] = matrix
        elif isinstance(matrix, np.ndarray):
            dense_arrays[key] = matrix
        elif matrix is None:
            print(f"{key} is None object. Skipping.")
            continue
        else:
            raise TypeError(f"Unsupported format {type(matrix)}.")

    # Save numpy arrays into a single file
    np.savez_compressed(f"{prefix}.npz", **dense_arrays)
    print("Save all dense arrays to",
          f"{prefix}.npz, including {list(dense_arrays.keys())}")

    # Save scipy sparse matrices into different files by keys
    for key, matrix in sparse_arrays.items():
        sp.save_npz(f"{prefix}_{key}.sparse.npz", matrix)
        print("Save sparse matrix", key, "to", f"{prefix}_{key}.sparse.npz")
