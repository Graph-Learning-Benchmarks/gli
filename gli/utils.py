"""Utility functions."""
import contextlib
import itertools
import json
import os
import re
import subprocess
import warnings
from typing import Iterator, Optional, Tuple
from urllib.parse import urlparse
import hashlib

import dgl
import numpy as np
import requests
import scipy.sparse as sp
import torch
from torch.utils.model_zoo import tqdm

import gli.config
from gli import DATASET_PATH, GLOBAL_FILE_URL, ROOT_PATH, WARNING_DENSE_SIZE, \
    SERVER_IP


def get_available_datasets():
    """Get available datasets from GitHub.

    :return: List of available datasets.
    :rtype: List[str]
    """
    github_api_url = ("https://api.github.com/"
                      "repos/Graph-Learning-Benchmarks/gli/contents/datasets")
    response = requests.get(github_api_url, timeout=5)
    response = response.json()
    return [folder["name"] for folder in response if folder["type"] == "dir"]


def fetch_dataset(dataset_name: str):
    """Fetch (Download) a dataset from GitHub.

    Note
    ====
    The dataset will be downloaded to `~/.gli/datasets/` by default.
    """
    github_api_url = ("https://api.github.com/"
                      "repos/Graph-Learning-Benchmarks/gli/contents/datasets/"
                      f"{dataset_name}")
    response = requests.get(github_api_url, timeout=5)
    if response.status_code != 200:
        raise ValueError(f"Dataset {dataset_name} not found.")
    response = response.json()
    files = [file["name"] for file in response if file["type"] == "file"]
    if os.path.exists(f"{DATASET_PATH}/{dataset_name}"):
        warnings.warn(f"Dataset {dataset_name} already exists.")
        return
    else:
        os.makedirs(f"{DATASET_PATH}/{dataset_name}")
    for file in files:
        curl_cmd = (
            "curl -l https://raw.githubusercontent.com/"
            "graph-learning-benchmarks/gli/main/datasets/"
            f"{dataset_name}/{file} -o {DATASET_PATH}/{dataset_name}/{file}")
        subprocess.call(curl_cmd, shell=True)


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
    params = {"id": file_id, "export": "download"}
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


def _find_data_files_from_json_files(data_dir):
    """Traverse json files under dataset path and find dependent data files."""
    json_files = []
    for file in os.listdir(data_dir):
        if file.endswith(".json") and file != "urls.json":
            json_files.append(file)

    def _find_data_files_helper(data):
        """Get all entries in data with key `file` at all levels."""
        data_files = []
        if isinstance(data, dict):
            for key, value in data.items():
                if key == "file":
                    data_files.append(value)
                else:
                    data_files.extend(_find_data_files_helper(value))
        elif isinstance(data, list):
            for item in data:
                data_files.extend(_find_data_files_helper(item))
        return data_files

    # Get all dependent data files from json files.
    data_files = []
    for json_file in json_files:
        with open(os.path.join(data_dir, json_file), "r",
                  encoding="utf-8") as f:
            data = json.load(f)
            data_files.extend(_find_data_files_helper(data))
    # Deduplicate data_files.
    data_files = list(set(data_files))

    return data_files


def _get_url_from_server(data_file: str):
    """Get url for a specific data file from server."""
    resp = requests.request("GET",
                            f"{SERVER_IP}/api/get-url/{data_file}",
                            timeout=5)
    print(resp.url)
    resp = resp.json()
    if resp["message_type"] == "error":
        return None
    elif resp["message_type"] == "url":
        return resp["content"]
    else:
        return None


def download_data(dataset: str, verbose=False):
    """Download dependent data of a configuration (metadata/task) file.

    Args:
        dataset (str): Name of dataset.
        verbose (bool, optional): Defaults to False.
    """
    data_dir = os.path.join(get_local_data_dir(), dataset)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"cannot find dataset {dataset}.")

    # Get all required dependent data files from json files.
    data_files = _find_data_files_from_json_files(data_dir)
    exist_all_files = True
    for data_file_name in data_files:
        data_file_path = os.path.join(data_dir, data_file_name)
        if not os.path.exists(data_file_path):
            exist_all_files = False
            break
    if exist_all_files:
        if verbose:
            print("All data files already exist. Skip downloading.")
        return

    # First, get urls from EC2
    data_file_url_dict = {}
    url_retrieval_success = True  # Whether all urls are retrieved from server
    for data_file in data_files:
        data_file_url_dict[data_file] = _get_url_from_server(data_file)
        if data_file_url_dict[data_file] is None:
            url_retrieval_success = False

    if not url_retrieval_success:
        # Second, check the local urls.json file.
        url_file = os.path.join(data_dir, "urls.json")
        if os.path.exists(url_file):
            with open(url_file, "r", encoding="utf-8") as fp:
                url_dict = json.load(fp)
        else:
            # Third, try to download and check the global_urls.json file.
            global_url_file = os.path.join(
                get_local_data_dir(), "global_urls.json")
            _download(GLOBAL_FILE_URL, global_url_file, verbose=verbose)
            if os.path.exists(global_url_file):
                with open(global_url_file, "r", encoding="utf-8") as fp:
                    url_dict = json.load(fp)

    # Get urls for all required data files.
    for data_file in data_files:
        if data_file not in data_file_url_dict \
                or data_file_url_dict[data_file] is None:
            if data_file in url_dict:
                data_file_url_dict[data_file] = url_dict[data_file]
            else:
                raise FileNotFoundError(f"cannot find url for {data_file}.")

    for data_file_name, url in data_file_url_dict.items():
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


def save_data(prefix, save_dir=".", **kwargs):
    """Save arrays into numpy binary formats with unique identifiers.

    :param prefix: The prefix of the saved files. See below for details.
    :type prefix: str
    :param save_dir: The directory to save the files.
    :type save_dir: str
    :param kwargs: The arrays to be saved. The key will be used as key for
        dense arrays and will be used in filenames for sparse arrays.
    :type kwargs: dict[str, numpy.ndarray or scipy.sparse.matrix]

    Dense arrays (numpy) will be saved in the below format as a single file:
    <prefix>__<md5>.npz

    Sparse arrays (scipy) will be saved in the below format individually:
    <prefix>__<key>__<md5>.sparse.npz

    The MD5 hash is calculated from the content of the file.

    Example
    -------
    ```python
    >>> data = {
        "node_feats": node_feats,
        "node_class": node_class,
        "edge": edge,
        "node_list": node_list,
        "edge_list": edge_list
    }
    >>> save_data("cora", **data)
    {'node_class': {'file': 'cora__<md5>.npz', 'key': 'node_class'},
     'edge': {'file': 'cora__<md5>.npz', 'key': 'edge'},
     'node_list': {'file': 'cora__<md5>.npz', 'key': 'node_list'},
     'edge_list': {'file': 'cora__<md5>.npz', 'key': 'edge_list'},
     'node_feats': {'file': 'cora__node_feats__<md5>.sparse.npz'}}
    ```

    :return: a dictionary that maps the key to the location of the saved array.
    :rtype: dict
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

    key_to_loc = {}

    # Check if the save_dir exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    def _dir(filename):
        """Prepend save_dir to the file."""
        return os.path.join(save_dir, filename)

    # Save numpy arrays into a single file
    np.savez_compressed(_dir(f"{prefix}.npz"), **dense_arrays)
    with open(_dir(f"{prefix}.npz"), "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    # Rename the file to include the md5 hash
    os.rename(_dir(f"{prefix}.npz"), _dir(f"{prefix}__{md5}.npz"))

    key_to_loc.update({
        key: {
            "file": f"{prefix}__{md5}.npz",
            "key": key
        }
        for key in dense_arrays
    })

    # Save scipy sparse matrices into different files by keys
    for key, matrix in sparse_arrays.items():
        sp.save_npz(_dir(f"{prefix}__{key}.sparse.npz"), matrix)
        with open(_dir(f"{prefix}__{key}.sparse.npz"), "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        os.rename(_dir(f"{prefix}__{key}.sparse.npz"),
                  _dir(f"{prefix}__{key}__{md5}.sparse.npz"))
        key_to_loc[key] = {"file": f"{prefix}__{key}__{md5}.sparse.npz"}

    return key_to_loc


def get_local_data_dir():
    """Get the local data storage directory.

    The function is used to determine whether the gli module is cloned from
    GitHub or downloaded by pypi. If the module is cloned from GitHub, the
    local data storage directory is ``./datasets``. Otherwise, the local data
    storage directory is ``~/.gli/datasets``.

    Note
    ----
    In the future, we will support git partial checkout to download only the
    source code of gli by git. In this case, the local data storage directory
    will still be ``./datasets`` but the directory will be empty.

    :return: gli's local data storage directory
    :rtype: str
    """
    # The repo is cloned to the local file system and is complete
    # In this case, we use the ./datasets folder
    full_repo_path = os.path.join(ROOT_PATH, "datasets")
    if os.path.exists(full_repo_path):
        return full_repo_path

    # The repo is installed by pypi and is incomplete
    # In this case, we use the ~/.gli/datasets folder
    if not os.path.exists(gli.config.DATASET_PATH):
        os.makedirs(gli.config.DATASET_PATH)
    return gli.config.DATASET_PATH
