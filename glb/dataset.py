"""Dataset for GLB."""
import json
import os

import torch
from dgl import DGLGraph
from dgl.data import DGLDataset

from glb.utils import load_data

SUPPORT_TASK_TYPES = ["NodeClassification"]


class GLBTask:
    """GLB task base class."""

    def __init__(self, task_dict, pwd):
        """Initialize GLBTask."""
        self.pwd = pwd
        self.type = task_dict["type"]
        self.description = task_dict["description"]
        self.features = task_dict["feature"]
        self.target = task_dict["target"]
        self.split = {
            "train_set": None,
            "val_set": None,
            "test_set": None
        }

        self._load(task_dict)

    def _load(self, task_dict):
        file_buffer = {}
        for dataset_ in self.split:
            filename = task_dict[dataset_]["file"]
            key = task_dict[dataset_].get("key")
            if filename not in file_buffer:
                file_buffer[filename] = load_data(
                    os.path.join(self.pwd, filename))
            indices = file_buffer[filename][key] if key else \
                file_buffer[filename]
            self.split[dataset_] = indices
            # indices can be mask tensor or an index tensor


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


def read_glb_task(task_path: os.PathLike, verbose=True):
    """Initialize and return a Task object given task_path."""
    pwd = os.path.dirname(task_path)
    with open(task_path, "r", encoding="utf-8") as fptr:
        task_dict = json.load(fptr)
    if verbose:
        print(task_dict["description"])
    if task_dict["type"] not in SUPPORT_TASK_TYPES:
        raise NotImplementedError

    return GLBTask(task_dict, pwd)
