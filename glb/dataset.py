"""Dataset for GLB."""
import torch
from dgl import DGLGraph
from dgl.data import DGLDataset

from glb.task import GLBTask


def node_classification_dataset_factory(graph: DGLGraph, task: GLBTask):
    """Initialize and return a NodeClassification Dataset."""
    if len(task.features) > 1:
        raise NotImplementedError("Only support single feature currently.")

    class NodeClassificationDataset(DGLDataset):
        """Node classification dataset."""

        def __init__(self):
            super().__init__(name=task.description)
            self._g = None
            self.features = task.features
            self.target = task.target
            self._num_labels = task.target["num_classes"]

        def process(self):
            self._g = graph
            for dataset_, indices_ in task.split.items():
                assert dataset_ not in self._g.ndata
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
