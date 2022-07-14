"""Dataset for GLB."""
from typing import Iterable
import numpy as np

import torch
from dgl import DGLGraph
from dgl.data import DGLDataset

from glb.task import (GraphClassificationTask, LinkPredictionTask,
                      NodeClassificationTask)


def node_classification_dataset_factory(graph: DGLGraph,
                                        task: NodeClassificationTask):
    """Initialize and return a NodeClassification Dataset."""
    assert isinstance(graph, DGLGraph)

    class NodeClassificationDataset(DGLDataset):
        """Node classification dataset."""

        def __init__(self):
            """Node classification dataset."""
            self._g = None
            self.features = task.features
            self.target = task.target
            self._num_labels = task.num_classes
            self.num_folds = task.num_folds
            super().__init__(name=task.description, force_reload=True)

        def process(self):
            """Add train, val, and test masks to graph."""
            self._g = graph
            for dataset_, indices_list_ in task.split.items():
                mask_list = []
                for fold in range(self.num_folds):
                    if self.num_folds == 1:
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
                if self.num_folds == 1:
                    mask = mask_list[0]
                else:
                    mask = torch.stack(mask_list, dim=1)
                self._g.ndata[dataset_.replace("set", "mask")] = mask.bool()

        def __getitem__(self, idx):
            assert idx == 0, "This dataset has only one graph"
            return self._g

        def __len__(self):
            return 1

        @property
        def num_labels(self):
            return self._num_labels

    return NodeClassificationDataset()


def graph_classification_dataset_factory(graphs: Iterable[DGLGraph],
                                         task: GraphClassificationTask):
    """Initialize and return split GraphClassification Dataset."""
    assert isinstance(graphs, Iterable)

    entries = task.target.split("/")
    assert len(entries) == 2
    assert entries[0] == "Graph"
    label_name = entries[1]

    class GraphClassificationDataset(DGLDataset):
        """Graph Classification Dataset."""

        def __init__(self, split="train_set"):
            self.graphs = None
            self.features = task.features
            self.target = task.target
            self._num_labels = task.num_classes
            self.split = split

            if task.num_folds > 1:
                raise NotImplementedError(
                    "GraphClassificationDataset does not support multi-split.")

            super().__init__(name=task.description, force_reload=True)

        def process(self):
            """Add train, val, and test masks to graph."""
            self.graphs = graphs
            device = graphs[0].device
            indices = task.split[self.split]
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
            return self.graphs[idx], getattr(self.graphs[idx], label_name)

        def __len__(self):
            return len(self.graphs)

        @property
        def num_labels(self):
            return self._num_labels

    datasets = []
    for split in task.split:
        datasets.append(GraphClassificationDataset(split=split))

    return datasets


def link_prediction_dataset_factory(graph: DGLGraph, task: LinkPredictionTask):
    """Initialize and return a LinkPrediction Dataset."""
    assert isinstance(graph, DGLGraph)

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
