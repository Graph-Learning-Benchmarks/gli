"""Dataset for GLB."""
from typing import Iterable
import numpy as np

import torch
from dgl.data import DGLDataset

from glb.task import (GLBTask, GraphClassificationTask, GraphRegressionTask,
                      LinkPredictionTask, NodeClassificationTask,
                      NodeRegressionTask)
from glb.graph import GLBGraph


class NodeDataset(DGLDataset):
    """Node level dataset."""

    def __init__(self, graph: GLBGraph, task: GLBTask):
        """Initialize a NodeDataset with a graph and a task.

        Args:
            graph (GLBGraph): A GLB graph.
            task (GLBTask): Node level GLB task config.
        """
        self._g = graph
        self.features = task.features
        self.target = task.target
        self.num_splits = task.num_splits
        self.task = task
        super().__init__(name=task.description, force_reload=True)

    def process(self):
        """Add train, val, and test masks to graph."""
        for dataset_, indices_list_ in self.task.split.items():
            mask_list = []
            for fold in range(self.num_splits):
                if self.num_splits == 1:
                    indices_ = indices_list_
                else:
                    indices_ = indices_list_[fold]

                assert not indices_.is_sparse
                indices_ = indices_.to(self._g.device)
                indices_ = torch.squeeze(indices_)
                assert indices_.dim() == 1
                if len(indices_) < self._g.num_nodes():  # index tensor
                    mask = torch.zeros(self._g.num_nodes(),
                                       device=self._g.device)
                    mask[indices_] = 1
                else:
                    mask = indices_
                mask_list.append(mask)
            if self.num_splits == 1:
                mask = mask_list[0]
            else:
                mask = torch.stack(mask_list, dim=1)
            self._g.ndata[dataset_.replace("set", "mask")] = mask.bool()

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        return 1


class NodeClassificationDataset(NodeDataset):
    """Node classification dataset."""
    def __init__(self, graph: GLBGraph, task: NodeClassificationTask):
        """Initialize a node classification dataset with a graph and a task.

        Args:
            graph (GLBGraph): A GLB graph
            task (NodeClassificationTask): Node classification task config.
        """
        super().__init__(graph, task)
        self.num_labels = task.num_classes


class NodeRegressionDataset(NodeDataset):
    """Node regression dataset."""
    def __init__(self, graph: GLBGraph, task: NodeRegressionTask):
        """Initialize a node regression dataset with a graph and a task.

        Args:
            graph (GLBGraph): A GLB graph
            task (NodeRegressionTask): Node regression task config.
        """
        super().__init__(graph, task)


def node_dataset_factory(graph: GLBGraph, task: GLBTask):
    """Initialize and return a NodeDataset.
    Args:
        graph (GLBGraph): A GLB Graph
        task (GLBTask): GLB task config

    Raises:
        TypeError: Unknown task type

    Returns:
        NodeDataset: Node level dataset
    """
    assert isinstance(graph, GLBGraph)
    if isinstance(task, NodeRegressionTask):
        return NodeRegressionDataset(graph, task)
    elif isinstance(task, NodeClassificationTask):
        return NodeClassificationDataset(graph, task)
    else:
        raise TypeError(f"Unknown task type {type(task)}")


class GraphDataset(DGLDataset):
    """Graph Dataset."""

    def __init__(self,
                 graphs: Iterable[GLBGraph],
                 task: GLBTask,
                 split="train_set"):
        """Initialize a graph level dataset.

        Args:
            graphs (Iterable[GLBGraph]): A list of GLB graphs
            task (GLBTask): GLB task config
            split (str, optional): One of "train_set", "test_set", and
                "val_set". Defaults to "train_set".

        Raises:
            NotImplementedError: GraphDataset does not support multi-split.
        """
        self.graphs = graphs
        self.features = task.features
        self.target = task.target
        self.split = split
        self.label_name = None
        self.task = task

        if task.num_splits > 1:
            raise NotImplementedError(
                "GraphDataset does not support multi-split.")

        super().__init__(name=task.description, force_reload=True)

    def process(self):
        """Add train, val, and test masks to graph."""
        entries = self.task.target.split("/")
        assert len(entries) == 2
        assert entries[0] == "Graph"
        self.label_name = entries[1]

        device = self.graphs[0].device
        indices = self.task.split[self.split]
        assert not indices.is_sparse
        if isinstance(indices, np.ndarray):
            indices = torch.from_numpy(indices).to(device)
        else:
            indices = indices.to(device)
        indices = torch.squeeze(indices)
        assert indices.dim() == 1
        if len(indices) < len(self.graphs):  # index tensor
            mask = torch.zeros(len(self.graphs))
            mask[indices] = 1
        else:
            mask = indices

        graphs_in_split = []
        for is_in_split, g in zip(mask, self.graphs):
            if is_in_split:
                graphs_in_split.append(g)

        self.graphs = graphs_in_split

    def __getitem__(self, idx):
        return self.graphs[idx], getattr(self.graphs[idx], self.label_name)

    def __len__(self):
        return len(self.graphs)


class GraphClassificationDataset(GraphDataset):
    """Graph classification dataset."""
    def __init__(self,
                 graphs: Iterable[GLBGraph],
                 task: GraphClassificationTask,
                 split="train_set"):
        super().__init__(graphs, task, split)
        """Initialize a graph classification dataset."""
        self.num_labels = task.num_classes


class GraphRegressionDataset(GraphDataset):
    """Graph regression dataset."""
    def __init__(self,
                 graphs: Iterable[GLBGraph],
                 task: GraphRegressionTask,
                 split="train_set"):
        """Initialize a graph regression dataset."""
        super().__init__(graphs, task, split)


def graph_classification_dataset_factory(graphs: Iterable[GLBGraph],
                                         task: GLBTask):
    """Initialize and return split GraphDataset.

    Args:
        graphs (Iterable[GLBGraph]): A list of GLB graphs.
        task (GLBTask): GLB task config

    Raises:
        TypeError: Unknown task type.

    Returns:
        GraphDataset: Graph level dataset.
    """
    assert isinstance(graphs, Iterable)

    datasets = []
    for split in task.split:
        if isinstance(task, GraphClassificationTask):
            datasets.append(
                GraphClassificationDataset(graphs, task, split=split))
        elif isinstance(task, GraphRegressionTask):
            datasets.append(GraphRegressionDataset(graphs, task, split=split))
        else:
            raise TypeError(f"Unknown task type {type(task)}.")

    return datasets


def link_prediction_dataset_factory(graph: GLBGraph, task: LinkPredictionTask):
    """Initialize and return a LinkPrediction Dataset."""
    assert isinstance(graph, GLBGraph)

    class LinkPredictionDataset(DGLDataset):
        """Link Prediction dataset."""

        def __init__(self):
            self._g = None
            self.features = task.features
            self.target = task.target
            self.sample_runtime = task.sample_runtime
            super().__init__(name=task.description, force_reload=True)

        def process(self):
            self._g = graph
            if task.type == "TimeDependentLinkPrediction":
                # load train, val, test edges
                time_entries = task.time.split("/")
                assert len(time_entries) == 2
                assert time_entries[0] == "Edge"
                time_attr = time_entries[-1]
                etime = self._g.edata[time_attr]  # tensor / dict of tensor
                for split in ["train", "valid", "test"]:
                    window = task.time_window[f"{split}_time_window"]
                    if isinstance(etime, dict):
                        self._g.edata[f"{split}_mask"] = {
                            k: torch.logical_and(v >= window[0], v < window[1])
                            for k, v in etime.items()
                        }
                    else:
                        self._g.edata[f"{split}_mask"] = torch.logical_and(
                            etime >= window[0], etime < window[1])
            else:
                raise NotImplementedError

        def get_idx_split(self):
            split_dict = {}
            for split in ["train", "valid", "test"]:
                split_dict[split] = torch.masked_select(
                    torch.arange(self._g.num_edges()),
                    self._g.edata[f"{split}_mask"])
            return split_dict

    return LinkPredictionDataset()
