"""Dataset for GLB."""
from typing import Iterable
import numpy as np

import torch
from dgl.data import DGLDataset
from dgl import DGLGraph

from glb.task import (EntityLinkPredictionTask, GLBTask,
                      GraphClassificationTask, GraphRegressionTask,
                      LinkPredictionTask, NodeClassificationTask,
                      NodeRegressionTask, RelationLinkPredictionTask,
                      TimeDependentLinkPredictionTask)


class GLBDataset(DGLDataset):
    
    def __init__(self, task: GLBTask):
        self.features = task.features
        self.target = task.target
        self.num_splits = task.num_splits
        self.split = task.split

        super().__init__(name=task.description, force_reload=True)


class NodeDataset(GLBDataset):
    """Node level dataset."""

    def __init__(self, graph: DGLGraph, task: GLBTask):
        """Initialize a NodeDataset with a graph and a task.

        Args:
            graph (DGLGraph): A DGL graph.
            task (GLBTask): Node level GLB task config.
        """
        self._g = graph
        super().__init__(task)

    def process(self):
        """Add train, val, and test masks to graph."""
        for dataset_, indices_list_ in self.split.items():
            mask_list = []
            for fold in range(self.num_splits):
                if self.num_splits == 1:
                    indices_ = indices_list_
                else:
                    indices_ = indices_list_[fold]

                assert not (indices_.is_sparse or indices_.is_sparse_csr)
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
        """Single graph dataset only has 1 element."""
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        """Single graph dataset only has 1 element."""
        return 1


class NodeClassificationDataset(NodeDataset):
    """Node classification dataset."""

    def __init__(self, graph: DGLGraph, task: NodeClassificationTask):
        """Initialize a node classification dataset with a graph and a task.

        Args:
            graph (DGLGraph): A DGL graph
            task (NodeClassificationTask): Node classification task config.
        """
        super().__init__(graph, task)
        self.num_labels = task.num_classes


class NodeRegressionDataset(NodeDataset):
    """Node regression dataset."""

    def __init__(self, graph: DGLGraph, task: NodeRegressionTask):
        """Initialize a node regression dataset with a graph and a task.

        Args:
            graph (DGLGraph): A DGL graph
            task (NodeRegressionTask): Node regression task config.
        """
        super().__init__(graph, task)


class GraphDataset(GLBDataset):
    """Graph Dataset."""

    def __init__(self,
                 graphs: Iterable[DGLGraph],
                 task: GLBTask,
                 split_set="train_set"):
        """Initialize a graph level dataset.

        Args:
            graphs (Iterable[DGLGraph]): A list of GLB graphs
            task (GLBTask): GLB task config
            split (str, optional): One of "train_set", "test_set", and
                "val_set". Defaults to "train_set".

        Raises:
            NotImplementedError: GraphDataset does not support multi-split.
        """
        self.graphs = graphs
        self.label_name = None
        self.split_set = split_set

        if task.num_splits > 1:
            raise NotImplementedError(
                "GraphDataset does not support multi-split.")

        super().__init__(task)

    def process(self):
        """Add train, val, and test masks to graph."""
        entries = self.target.split("/")
        assert len(entries) == 2
        assert entries[0] == "Graph"
        self.label_name = entries[1]

        device = self.graphs[0].device
        indices = self.split[self.split_set]
        assert not (indices.is_sparse or indices.is_sparse_csr)
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
        """Get a pair of graph and label."""
        return self.graphs[idx], getattr(self.graphs[idx], self.label_name)

    def __len__(self):
        """Magic method."""
        return len(self.graphs)


class GraphClassificationDataset(GraphDataset):
    """Graph classification dataset."""

    def __init__(self,
                 graphs: Iterable[DGLGraph],
                 task: GraphClassificationTask,
                 split="train_set"):
        """Initialize a graph classification dataset."""
        super().__init__(graphs, task, split)
        self.num_labels = task.num_classes


class GraphRegressionDataset(GraphDataset):
    """Graph regression dataset."""

    def __init__(self,
                 graphs: Iterable[DGLGraph],
                 task: GraphRegressionTask,
                 split="train_set"):
        """Initialize a graph regression dataset."""
        super().__init__(graphs, task, split)


class EdgeDataset(GLBDataset):
    """Edge level dataset."""

    def __init__(self, graph: DGLGraph, task: GLBTask):
        """Initialize a edge level dataset.

        Args:
            graph (DGLGraph): A DGL graph
            task (GLBTask): GLB task config

        Raises:
            NotImplementedError: GraphDataset does not support multi-split.
        """
        self._g = graph
        self.sample_runtime = task.sample_runtime
        super().__init__(task)

    def process(self):
        for split in ("train", "val", "test"):
            indices = torch.zeros(self._g.num_edges(), dtype=torch.bool)
            indices[self.split[f"{split}_set"]] = True
            self._g.edata[f"{split}_mask"] = indices


class LinkPredictionDataset(EdgeDataset):

    def get_idx_split(self):
        """Return a dictionary of train, val, and test splits.

        Returns:
            Dict: split_dict
        """
        split_dict = {}
        for split in ("train", "val", "test"):
            split_dict[split] = torch.masked_select(
                torch.arange(self._g.num_edges()),
                self._g.edata[f"{split}_mask"])
        return split_dict

    def __getitem__(self, idx):
        """Single graph dataset only has 1 element."""
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        """Single graph dataset only has 1 element."""
        return 1


class TimeDependentLinkPredictionDataset(LinkPredictionDataset):
    """Link Prediction dataset."""

    def __init__(self, graph: DGLGraph, task: GLBTask):
        """Initialize a edge level dataset.

        Args:
            graph (DGLGraph): A DGL graph
            task (GLBTask): GLB task config

        Raises:
            NotImplementedError: GraphDataset does not support multi-split.
        """
        self.time = task.time
        self.time_window = task.time_window
        super().__init__(graph, task)

    def process(self):
        time_entries = self.time.split("/")
        assert len(time_entries) == 2
        assert time_entries[0] == "Edge"
        time_attr = time_entries[-1]
        etime = self._g.edata[time_attr].squeeze()
        for split in ("train", "val", "test"):
            window = self.time_window[f"{split}_time_window"]
            self.split[f"{split}_set"] = torch.arange(
                self._g.num_edges())[torch.logical_and(etime >= window[0],
                                                       etime < window[1])]
        super().process()


class EntityLinkPredictionDataset(LinkPredictionDataset):

    pass


class RelationLinkPredictionDatset(LinkPredictionDataset):

    pass


def node_dataset_factory(graph: DGLGraph, task: GLBTask):
    """Initialize and return a NodeDataset.

    Args:
        graph (DGLGraph): A DGL graph
        task (GLBTask): GLB task config

    Raises:
        TypeError: Unknown task type

    Returns:
        NodeDataset: Node level dataset
    """
    assert isinstance(graph, DGLGraph)
    if isinstance(task, NodeRegressionTask):
        return NodeRegressionDataset(graph, task)
    elif isinstance(task, NodeClassificationTask):
        return NodeClassificationDataset(graph, task)
    else:
        raise TypeError(f"Unknown task type {type(task)}")


def edge_dataset_factory(graph: DGLGraph, task: LinkPredictionTask):
    """Initialize and return a LinkPrediction Dataset."""
    assert isinstance(graph, DGLGraph)

    if isinstance(task, TimeDependentLinkPredictionTask):
        return TimeDependentLinkPredictionDataset(graph, task)
    elif isinstance(task, RelationLinkPredictionTask):
        return RelationLinkPredictionDatset(graph, task)
    elif isinstance(task, EntityLinkPredictionTask):
        return EntityLinkPredictionDataset(graph, task)
    elif isinstance(task, LinkPredictionTask):
        return LinkPredictionDataset(graph, task)
    else:
        raise TypeError(f"Unknown task type {type(task)}.")


def graph_dataset_factory(graphs: Iterable[DGLGraph], task: GLBTask):
    """Initialize and return split GraphDataset.

    Args:
        graphs (Iterable[DGLGraph]): A list of GLB graphs.
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
