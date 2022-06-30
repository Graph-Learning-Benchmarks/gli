"""Data loading module for user-side."""
import os
from typing import List, Union

from dgl import DGLGraph

import glb.dataset
from glb import ROOT_PATH
from glb.graph import read_glb_graph
from glb.task import GLBTask, read_glb_task
from glb.utils import download_data


def combine_graph_and_task(graph: Union[DGLGraph, List[DGLGraph]],
                           task: GLBTask):
    """Return a dataset given graph and task."""
    if task.type == "NodeClassification":
        return glb.dataset.node_classification_dataset_factory(graph, task)
    elif task.type == "GraphClassification":
        return glb.dataset.graph_classification_dataset_factory(graph, task)
    elif task.type in ["TimeDependentLinkPrediction", "LinkPrediction"]:
        return glb.dataset.link_prediction_dataset_factory(graph, task)
    raise NotImplementedError


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
    """Get a know GLB task of a given dataset.

    Args:
        dataset (str): Name of dataset.
        task (str): Name of task.
        verbose (bool, optional): Defaults to True.

    Returns:
        GLBTask: Task object that represents the predefined task.
    """
    data_dir = os.path.join(ROOT_PATH, "datasets/", dataset)
    task_path = os.path.join(data_dir, f"{task}.json")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"{data_dir} not found.")
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"{task_path} not found.")
    download_data(dataset, verbose=verbose)

    return read_glb_task(task_path, verbose=verbose)
