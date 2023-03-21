"""The ``gli.dataloading`` module provides functions to load graph datasets."""
import os
from typing import List, Union

from dgl import DGLGraph
from dgl.data import DGLDataset

import gli.dataset
from gli.graph import read_gli_graph
from gli.task import GLITask, read_gli_task
from gli.utils import download_data, fetch_dataset, get_local_data_dir


def combine_graph_and_task(graph: Union[DGLGraph, List[DGLGraph]],
                           task: GLITask) -> DGLDataset:
    """Combine graph(s) and task to get a dataset.

    :func:`gli.dataloading.combine_graph_and_task` loads task-specific
    information into the inputed graph(s) as additional attributes. The
    graph(s) is then further wrapped by a :class:`dgl.data.DGLDataset` object.

    :param graph: graph or a list of graphs.
    :type graph: :class:`dgl.DGLGraph` or list of :class:`dgl.DGLGraph`.
    :param task: predefined task configuration.
    :type task: :class:`gli.task.GLITask`.

    :rtype: :class:`dgl.data.DGLDataset`.

    Example
    -------
    .. code-block:: python

        >>> g = get_gli_graph(dataset="cora")
        >>> t = get_gli_task(dataset="cora", task="NodeClassification")
        >>> d = combine_graph_and_task(g, t)
        >>> d.name
        'CORA dataset. NodeClassification'
    """
    if task.type in ("NodeClassification", "NodeRegression"):
        return gli.dataset.node_dataset_factory(graph, task)
    elif task.type in ("GraphClassification", "GraphRegression"):
        return gli.dataset.graph_dataset_factory(graph, task)
    elif task.type in ("TimeDependentLinkPrediction", "LinkPrediction",
                       "KGEntityPrediction", "KGRelationPrediction"):
        return gli.dataset.edge_dataset_factory(graph, task)
    raise NotImplementedError(f"Unsupported type {task.type}")


def get_gli_dataset(dataset: str,
                    task: str,
                    task_id: int = 1,
                    device: str = "cpu",
                    verbose: bool = False) -> DGLDataset:
    """Get a graph dataset given dataset name and task config.

    :param dataset: graph dataset name.
    :type dataset: str
    :param task: task type, e.g. "NodeClassification", "NodeRegression".
    :type task: str
    :param task_id: task ID defined in dataset folder, defaults to 1.
    :type task_id: int, optional.
    :param device: device name, defaults to "cpu".
    :type device: str, optional
    :param verbose: verbose level, defaults to False.
    :type verbose: bool, optional

    :rtype: :class:`dgl.data.DGLDataset`.

    This function essentially performs the following steps:

    .. code-block:: python

        g = get_gli_graph(dataset, device=device, verbose=verbose) t =
        get_gli_task(dataset, task, task_id=task_id, verbose=verbose) return
        combine_graph_and_task(g, t)

    .. note::
        :func:`gli.dataloading.get_gli_dataset` will download the data files
        if the data files do not exist in the local file system.

    Examples
    --------
    .. code-block:: python

        >>> d = get_gli_dataset(dataset="cora", task="NodeClassification")
        >>> d.name
        'CORA dataset. NodeClassification'
    """
    g = get_gli_graph(dataset, device=device, verbose=verbose)
    t = get_gli_task(dataset, task, task_id=task_id, verbose=verbose)
    return combine_graph_and_task(g, t)


def get_gli_graph(dataset: str,
                  device: str = "cpu",
                  verbose: bool = False) -> Union[DGLGraph, List[DGLGraph]]:
    """Get one (or a list of) :class:`dgl.DGLGraph` object(s) from GLI repo.

    If the loaded graph dataset contains a single graph, the returned value is
    a single :class:`dgl.DGLGraph` object. Otherwise, if the dataset contains
    multiple graphs, the returned value is a list of :class:`dgl.DGLGraph`
    objects.

    :param dataset: graph dataset name.
    :type dataset: str
    :param device: device name, defaults to "cpu".
    :type device: str, optional
    :param verbose: verbose level, defaults to False.
    :type verbose: bool, optional

    :rtype: :class:`dgl.DGLGraph` or list of :class:`dgl.DGLGraph`.

    :raises FileNotFoundError: Raised when metadata/task configuration file is
        not found.

    .. note::
        :func:`gli.dataloading.get_gli_graph` will download the data files if
        the data files do not exist in the local file system.

    Examples
    --------
    .. code-block:: python

        >>> g = gli.get_gli_graph("cora")
        >>> g
        Graph(num_nodes=2708, num_edges=10556,
            ndata_schemes={...}
            edata_schemes={})
    """
    data_dir = get_local_data_dir()
    if data_dir == gli.config.DATASET_PATH:
        fetch_dataset(dataset)
    data_dir = os.path.join(data_dir, dataset)
    metadata_path = os.path.join(data_dir, "metadata.json")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"{data_dir} not found.")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"{metadata_path} not found.")
    download_data(dataset, verbose=verbose)

    return read_gli_graph(metadata_path, device=device, verbose=verbose)


def get_gli_task(dataset: str,
                 task: str,
                 task_id: int = 1,
                 verbose: bool = False) -> GLITask:
    """Get a GLI task configuration object from GLI repo.

    The returned :class:`gli.task.GLITask` object is an intermediate product of
    the dataloading pipeline. It contains the task-specific information.

    :param dataset: graph dataset name.
    :type dataset: str
    :param task: task type, e.g. "NodeClassification", "NodeRegression".
    :type task: str
    :param task_id: task ID defined in dataset folder, defaults to 1.
    :type task_id: int, optional.
    :param verbose: verbose level, defaults to False.
    :type verbose: bool, optional

    :rtype: :class:`gli.task.GLITask`

    :raise NotImplementedError: Raised when task type is unsupported.
    :raise FileNotFoundError: Raised when task configuration file is not found.

    Examples
    --------
    .. code-block:: python

        >>> get_gli_task(dataset="cora", task="NodeClassification", task_id=1)
        Node classification on CORA dataset. Planetoid split.
        <gli.task.NodeClassificationTask object at 0x...>
    """
    name_map = {
        "NodeClassification": "node_classification",
        "GraphClassification": "graph_classification",
        "LinkPrediction": "link_prediction",
        "TimeDependentLinkPrediction": "time_dependent_link_prediction",
        "KGRelationPrediction": "kg_relation_prediction",
        "KGEntityPrediction": "kg_entity_prediction",
        "GraphRegression": "graph_regression",
        "NodeRegression": "node_regression"
    }
    if task not in name_map:
        raise NotImplementedError(f"Unsupported task type {task}.")
    data_dir = get_local_data_dir()
    if data_dir == gli.config.DATASET_PATH:
        fetch_dataset(dataset)
    data_dir = os.path.join(data_dir, dataset)
    task_path = os.path.join(data_dir, f"task_{name_map[task]}_{task_id}.json")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"{data_dir} not found.")
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"{task_path} not found.")
    download_data(dataset, verbose=verbose)

    return read_gli_task(task_path, verbose=verbose)
