"""Data loading module for user-side."""
import os
from typing import List, Union

from dgl import DGLGraph
from dgl.data import DGLDataset

import gli.dataset
from gli import ROOT_PATH
from gli.graph import read_gli_graph
from gli.task import GLITask, read_gli_task
from gli.utils import download_data


def combine_graph_and_task(graph: Union[DGLGraph, List[DGLGraph]],
                           task: GLITask) -> DGLDataset:
    """Combine graph(s) and task to get a dataset.

    Parameters
    ----------
    graph : Union[DGLGraph, List[DGLGraph]]
        Graph or a list of graphs.
    task : GLITask
        Predefined task configuration.

    Returns
    -------
    DGLDataset
        Graph dataset instance.

    Raises
    ------
    NotImplementedError
        Raised when task types are unknown.

    Examples
    --------
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

    GLI will download the dataset if the data files do not exist.

    Parameters
    ----------
    dataset : str
        Dataset/Graph name
    task : str
        Task type
    task_id : int, optional
        Task id defined in dataset folder, by default 1
    device : str, optional
        Device name, by default "cpu"
    verbose : bool, optional
        Verbose level, by default False

    Returns
    -------
    DGLDataset
        A DGL dataset instance

    Examples
    --------
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
    # pylint: disable=line-too-long
    """Get a GLI graph object, or a list of GLI graph objects.

    If the metadata defines multiple subgraphs on the dataset, the returned
    value is a list rather than a single graph.

    Parameters
    ----------
    dataset : str
        Dataset/Graph name
    device : str, optional
        Task type, by default "cpu"
    verbose : bool, optional
        Verbose level, by default False

    Returns
    -------
    Union[DGLGraph, List[DGLGraph]]
        Graph dataset instance

    Raises
    ------
    FileNotFoundError
        Raised when metadata/task configuration file is not found.

    Examples
    --------
    >>> g = get_gli_graph(dataset="cora")
    >>> g
    Graph(num_nodes=2708, num_edges=10556,
        ndata_schemes={'NodeFeature': Scheme(shape=(1433,), dtype=torch.float32), 'NodeLabel': Scheme(shape=(), dtype=torch.int64)}
        edata_schemes={})

    Notes
    -----
    The returned graph(s) is essentially DGLGraph with extra attributes defined
    by GLI.
    """  # noqa: E501
    data_dir = os.path.join(ROOT_PATH, "datasets/", dataset)
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
    """Get a GLI task configuration object.

    Parameters
    ----------
    dataset : str
        Dataset/Graph name
    task : str
        Task type
    task_id : int, optional
        Task id defined in dataset folder, by default 1
    verbose : bool, optional
        Verbose level, by default False

    Returns
    -------
    GLITask
        A GLI task configuration

    Raises
    ------
    NotImplementedError
        Raised when task type is unsupported.
    FileNotFoundError
        Raised when metadata/task configuration file is not found.

    Examples
    --------
    >>> get_gli_task(dataset="cora", task="NodeClassification", task_id=1)
    Node classification on CORA dataset. Planetoid split.
    <gli.task.NodeClassificationTask object at 0x100ad5760>
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
    data_dir = os.path.join(ROOT_PATH, "datasets/", dataset)
    task_path = os.path.join(data_dir, f"task_{name_map[task]}_{task_id}.json")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"{data_dir} not found.")
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"{task_path} not found.")
    download_data(dataset, verbose=verbose)

    return read_gli_task(task_path, verbose=verbose)
