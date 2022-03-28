"""Data loading module for user-side."""
from typing import Union, List
from dgl import DGLGraph

import glb.dataset
from glb.task import GLBTask


def combine_graph_and_task(graph: Union[DGLGraph, List[DGLGraph]],
                           task: GLBTask):
    """Return a dataset given graph and task."""
    if task.type == "NodeClassification":
        return glb.dataset.node_classification_dataset_factory(graph, task)
    elif task.type == "GraphClassification":
        return glb.dataset.graph_classification_dataset_factory(graph, task)
    raise NotImplementedError
