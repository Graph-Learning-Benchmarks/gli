"""Data loading module for user-side."""
import os
from typing import List, Union

import glb.dataset
from glb import ROOT_PATH
from glb.graph import read_glb_graph, GLBGraph
from glb.task import GLBTask, read_glb_task
from glb.utils import download_data


def combine_graph_and_task(graph: Union[GLBGraph, List[GLBGraph]],
                           task: GLBTask):
    """Combine graph and task to get a GLB dataset.

    Args:
        graph (Union[GLBGraph, List[GLBGraph]]): Graph(s) to construct dataset.
        task (GLBTask): GLB task config

    Raises:
        NotImplementedError: Unknown task type

    Returns:
        DGLDataset
    """
    if task.type in ("NodeClassification", "NodeRegression"):
        return glb.dataset.node_dataset_factory(graph, task)
    elif task.type == ("GraphClassification", "GraphRegression"):
        return glb.dataset.graph_dataset_factory(graph, task)
    elif task.type in ("TimeDependentLinkPrediction", ):
        return glb.dataset.edge_dataset_factory(graph, task)
    raise NotImplementedError


def get_glb_dataset(dataset: str, task: str, device="cpu", verbose=True):
    """Get a known GLB dataset of a given task.

    Args:
        dataset (str): Name of dataset.
        task (str): Name of task file.
        device (str, optional): Returned dataset's device. Defaults to "cpu".
        verbose (bool, optional): Defaults to True.

    Returns:
        Dataset: a iterable dataset of a given task.
    """
    g = get_glb_graph(dataset, device=device, verbose=verbose)
    t = get_glb_task(dataset, task, verbose=verbose)
    return combine_graph_and_task(g, t)


def get_glb_graph(dataset: str, device="cpu", verbose=True):
    """Get a known GLB graph.

    Download dependent files if needed.

    Args:
        dataset (str): Name of dataset.
        device (str, optional): Returned graph's device. Defaults to "cpu".
        verbose (bool, optional): Defaults to True.

    Returns:
        DGLHeteroGraph: Graph object(s) that represents the dataset.
    """
    data_dir = os.path.join(ROOT_PATH, "datasets/", dataset)
    metadata_path = os.path.join(data_dir, "metadata.json")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"{data_dir} not found.")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"{metadata_path} not found.")
    download_data(dataset, verbose=verbose)

    return read_glb_graph(metadata_path, device=device, verbose=verbose)


def get_glb_task(dataset: str, task: str, verbose=True):
    """Get a known GLB task of a given dataset.

    Args:
        dataset (str): Name of dataset.
        task (str): Name of task.
        verbose (bool, optional): Defaults to True.

    Returns:
        GLBTask: Predefined GLB task.
    """
    data_dir = os.path.join(ROOT_PATH, "datasets/", dataset)
    task_path = os.path.join(data_dir, f"{task}.json")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"{data_dir} not found.")
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"{task_path} not found.")
    download_data(dataset, verbose=verbose)

    return read_glb_task(task_path, verbose=verbose)
