"""Task for GLB."""
import json
import os
from typing import List

from glb.utils import file_reader

SUPPORT_TASK_TYPES = [
    "NodeClassification", "GraphClassification", "TimeDependentLinkPrediction"
]


class GLBTask:
    """GLB task base class."""

    def __init__(self, task_dict, pwd, device="cpu"):
        """Initialize GLBTask."""
        self.pwd = pwd
        self.type = task_dict["type"]
        self.description = task_dict["description"]
        self.features: List[str] = task_dict["feature"]
        self.target: str = None
        self.num_folds = 1
        self.split = {"train_set": None, "val_set": None, "test_set": None}
        self.device = device

        self._load(task_dict)

    def _load(self, task_dict):
        pass


class ClassificationTask(GLBTask):
    """Classification task."""

    def __init__(self, task_dict, pwd):
        """Initialize num_classes."""
        super().__init__(task_dict, pwd)
        self.num_classes = task_dict["num_classes"]
        self.target = task_dict["target"]

    def _load(self, task_dict):
        self.num_folds = task_dict.get("num_folds", 1)
        assert self.num_folds >= 1
        for dataset_ in self.split:
            filename = task_dict[dataset_]["file"]
            key = task_dict[dataset_].get("key")
            path = os.path.join(self.pwd, filename)
            if self.num_folds > 1:
                self.split[dataset_] = []
                for fold in range(self.num_folds):
                    assert key[-4:] == "FOLD", "split key not ending with FOLD"
                    this_fold_key = f"{key[:-4]}{fold}"
                    self.split[dataset_].append(
                        file_reader.get(path, this_fold_key, self.device))
                    # can be a list of mask tensors or index tensors
            else:
                self.split[dataset_] = file_reader.get(path, key, self.device)
                # can be a mask tensor or an index tensor


class NodeClassificationTask(ClassificationTask):
    """Node classification task, alias."""

    pass


class GraphClassificationTask(ClassificationTask):
    """Graph classification task, alias."""

    pass


class LinkPredictionTask(GLBTask):
    """Link prediction task."""

    def __init__(self, task_dict, pwd):
        """Link/Edge prediction."""
        self.target = "Edge/_Edge"
        super().__init__(task_dict, pwd)

    pass


class TimeDependentLinkPredictionTask(LinkPredictionTask):
    """Time dependent link prediction task."""

    def __init__(self, task_dict, pwd):
        """Time dependent link prediction task."""
        self.time = task_dict["time"]
        self.time_window = {
            "train_time_window": task_dict["train_time_window"],
            "valid_time_window": task_dict["valid_time_window"],
            "test_time_window": task_dict["test_time_window"]
        }
        self.valid_neg = task_dict.get("valid_neg", None)
        self.test_neg = task_dict.get("test_neg", None)
        super().__init__(task_dict, pwd)

    def _load(self, task_dict):
        for neg_idx in ["valid_neg", "test_neg"]:
            if getattr(self, neg_idx, None):
                filename = task_dict[neg_idx]["file"]
                key = task_dict[neg_idx].get("key")
                path = os.path.join(self.pwd, filename)
                indices = file_reader.get(path, key, self.device)
                setattr(self, neg_idx, indices)


def read_glb_task(task_path: os.PathLike, verbose=True):
    """Initialize and return a Task object given task_path."""
    pwd = os.path.dirname(task_path)
    with open(task_path, "r", encoding="utf-8") as fptr:
        task_dict = json.load(fptr)
    if verbose:
        print(task_dict["description"])

    if task_dict["type"] == "NodeClassification":
        return NodeClassificationTask(task_dict, pwd)
    elif task_dict["type"] == "GraphClassification":
        return GraphClassificationTask(task_dict, pwd)
    elif task_dict["type"] == "TimeDependentLinkPrediction":
        return TimeDependentLinkPredictionTask(task_dict, pwd)
    else:
        raise NotImplementedError(f"Unrecognized task: {task_dict['type']}"
                                  f"Supported tasks: {SUPPORT_TASK_TYPES}")
