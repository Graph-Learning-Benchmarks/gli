from dgl import DGLGraph

import glb.dataset
from glb.dataset import GLBTask


def combine_graph_and_task(graph: DGLGraph, task: GLBTask):
    if task.type == "NodeClassification":
        return glb.dataset.node_classification_dataset_factory(graph, task)
    raise NotImplementedError
