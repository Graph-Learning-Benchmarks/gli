from abc import abstractmethod
from asyncore import dispatcher_with_send
import json
import os

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from .graph import Graph
from .utils import load_data


def _get_feature(graph: Graph, obj: str, attr: str):
    obj = obj.lower()
    return getattr(graph, f"{obj}_attrs").get(attr)


def get_split_dataset(metadata_path: os.PathLike,
                      task_path: os.PathLike,
                      verbose=False):
    pwd = os.path.dirname(task_path)
    graph = Graph.load_graph(metadata_path)
    with open(task_path, "r", encoding="utf-8") as fptr:
        task = json.load(fptr)
    if verbose:
        print(task["description"])
    if task["type"] == "NodeClassification":
        if len(task["feature"]) > 1:
            raise NotImplementedError("Only support single feature currently.")
        feature = _get_feature(graph, task["feature"][0]["object"],
                               task["feature"][0]["attribute"])
        target = _get_feature(graph, task["target"]["object"],
                              task["target"]["attribute"])
        num_classes = task["target"]["num_classes"]
        info = {
            "feature": feature,
            "target": target,
            "num_classes": num_classes
        }
        file_buffer = {}
        data_split = []
        for ds in ["train_set", "val_set", "test_set"]:
            ds_info = info.copy()
            filename = task[ds]["file"]
            key = task[ds].get("key")
            if filename not in file_buffer:
                file_buffer[filename] = load_data(os.path.join(pwd, filename))
            ds_info["indices"] = file_buffer[filename][
                key] if key else file_buffer[filename]
            data_split.append(
                NodeClassificationDataset(graph, verbose=verbose, **ds_info))
            if verbose:
                split = ds[:-4].capitalize()
                num_samples = len(ds_info["indices"])
                print(f"  #{split}Samples: {num_samples}")
        return data_split
    else:
        raise NotImplementedError


class GLBDataset(Dataset):
    """An abstract class representing a GLB dataset.
    
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive."""

    def __init__(self, graph: Graph, verbose=False):
        """Initialize GLBDataset."""
        super().__init__()
        self.graph = graph
        self.verbose = verbose

    @abstractmethod
    def __getitem__(self, idx):
        """Gets the data object at index."""
        pass

    @abstractmethod
    def __len__(self):
        """The number of examples in the dataset."""
        pass

    def process(self):
        """Overwrite to realize your own logic of processing the input data."""
        raise NotImplementedError


class NodeClassificationDataset(GLBDataset):
    """Node classification dataset."""

    def __init__(self, graph: Graph, verbose=False, **kwargs):
        """Initialize dataset.
        
        Valid & required keyword arguments include
        1. ``feature``: List of Feature objects.
        2. ``target``: Single Feature object (label).
        3. ``indices``: node indices.
        4. ``num_classes``: number of classes.
        """
        super().__init__(graph, verbose=verbose)
        for key in kwargs:
            try:
                setattr(self, key, kwargs[key])
            except KeyError:
                raise ValueError(f"Missing kwarg {key}.")

        self.description = kwargs.get("description", None)

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError
        else:
            return getattr(
                self, "feature").data  # TODO - consider multi-feature cases

    def __len__(self):
        return 1
