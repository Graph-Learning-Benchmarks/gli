"""Dataset for GLB."""
from typing import List
import torch
from dgl import DGLGraph
from dgl.data import DGLDataset

from glb.task import NodeClassificationTask, GraphClassificationTask


def node_classification_dataset_factory(graph: DGLGraph,
                                        task: NodeClassificationTask):
    """Initialize and return a NodeClassification Dataset."""
    assert isinstance(graph, DGLGraph)

    class NodeClassificationDataset(DGLDataset):
        """Node classification dataset."""

        def __init__(self):
            self._g = None
            self.features = task.features
            self.target = task.target
            self._num_labels = task.num_classes
            super().__init__(name=task.description, force_reload=True)

        def process(self):
            """Add train, val, and test masks to graph."""
            self._g = graph
            for dataset_, indices_ in task.split.items():
                indices_ = torch.from_numpy(indices_).to(self._g.device)
                indices_ = torch.squeeze(indices_)
                assert indices_.dim() == 1
                if len(indices_) < self._g.num_nodes():  # index tensor
                    mask = torch.zeros(self._g.num_nodes())
                    mask[indices_] = 1
                else:
                    mask = indices_
                self._g.ndata[dataset_] = mask.bool()

        def __getitem__(self, idx):
            assert idx == 0, "This dataset has only one graph"
            return self._g

        def __len__(self):
            return 1

    return NodeClassificationDataset()


def graph_classification_dataset_factory(graphs: List[DGLGraph],
                                         task: GraphClassificationTask):
    """Initialize and return split GraphClassification Dataset."""
    assert isinstance(graphs, list)

    entries = task.target.split("/")
    assert len(entries) == 2
    assert entries[0] == "Graph"
    _label_name = entries[1]

    class GraphClassificationDataset(DGLDataset):
        """Graph Classification Dataset."""

        def __init__(self, split="train_set"):
            self.graphs = None
            self.features = task.features
            self.target = task.target
            self._num_labels = task.num_classes
            self.split = split

            super().__init__(name=task.description, force_reload=True)

        def process(self):
            """Add train, val, and test masks to graph."""
            self.graphs = graphs
            device = graphs[0].device
            indices = task.split[self.split]
            indices = torch.from_numpy(indices).to(device)
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
            return self.graphs[idx], getattr(self.graphs[idx], _label_name)

        def __len__(self):
            return len(self.graphs)

    datasets = []
    for split in task.split:
        datasets.append(GraphClassificationDataset(split=split))

    return datasets
