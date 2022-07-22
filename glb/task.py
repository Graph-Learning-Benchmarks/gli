"""Task for GLB."""
import json
import math
import os
import random
from typing import List

import torch

from glb.utils import file_reader

SUPPORTED_TASK_TYPES = [
    "NodeClassification", "NodeRegression", "GraphClassification",
    "GraphRegression", "TimeDependentLinkPrediction"
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
        self.num_splits = 1
        self.random_split = False
        self.split = {"train_set": None, "val_set": None, "test_set": None}
        self.device = device

        if "train_set" not in task_dict and \
                "train_time_window" not in task_dict:
            # use random split
            self.random_split = True

            assert "train_ratio" in task_dict
            train_ratio = task_dict["train_ratio"]
            val_ratio = task_dict["val_ratio"]
            test_ratio = task_dict["test_ratio"]
            assert train_ratio + val_ratio + test_ratio <= 1

            num_samples = task_dict["num_samples"]
            seed = task_dict.get("seed", None)

            self._set_random_split(train_ratio, val_ratio, test_ratio,
                                   num_samples, seed)

        self._load(task_dict)

    def _set_random_split(self,
                          train_ratio,
                          val_ratio,
                          test_ratio,
                          num_samples,
                          seed=None):
        # back up random state to avoid cross influence
        state = random.getstate()
        if seed is not None:
            random.seed(seed)

        indices = list(range(num_samples))
        random.shuffle(indices)
        train_boundary = math.floor(train_ratio * num_samples)
        val_boundary = math.floor((train_ratio + val_ratio) * num_samples)
        train_set = indices[:train_boundary]
        val_set = indices[train_boundary:val_boundary]
        test_set = indices[-math.ceil(test_ratio * num_samples):]

        # check that test set and train-val set have no overlap
        train_val_set = train_set + val_set
        assert len(set(train_val_set).intersection(set(test_set))) == 0

        self.split = {
            "train_set": torch.LongTensor(train_set).to(device=self.device),
            "val_set": torch.LongTensor(val_set).to(device=self.device),
            "test_set": torch.LongTensor(test_set).to(device=self.device)
        }

        # recover random state to avoid cross influence
        random.setstate(state)

    def _load(self, task_dict):
        pass

    def _load_split(self, task_dict):
        self.num_splits = task_dict.get("num_splits", 1)
        assert self.num_splits >= 1

        if self.random_split:  # use random split; pass loading
            assert self.num_splits == 1
            return

        for dataset_ in self.split:
            filename = task_dict[dataset_]["file"]
            key = task_dict[dataset_].get("key")
            path = os.path.join(self.pwd, filename)
            if self.num_splits > 1:
                self.split[dataset_] = []
                for fold in range(self.num_splits):
                    assert key[-4:] == "FOLD", "split key not ending with FOLD"
                    this_fold_key = f"{key[:-4]}{fold}"
                    self.split[dataset_].append(
                        file_reader.get(path, this_fold_key, self.device))
                    # can be a list of mask tensors or index tensors
            else:
                self.split[dataset_] = file_reader.get(path, key, self.device)
                # can be a mask tensor or an index tensor


class ClassificationTask(GLBTask):
    """Classification task."""

    def __init__(self, task_dict, pwd, device="cpu"):
        """Initialize num_classes."""
        super().__init__(task_dict, pwd, device)
        self.num_classes = task_dict["num_classes"]
        self.target = task_dict["target"]

    def _load(self, task_dict):
        self._load_split(task_dict)


class RegressionTask(GLBTask):
    """Regression task."""

    def __init__(self, task_dict, pwd, device="cpu"):
        """Initialize target."""
        super().__init__(task_dict, pwd, device)
        self.target = task_dict["target"]

    def _load(self, task_dict):
        self._load_split(task_dict)


class NodeClassificationTask(ClassificationTask):
    """Node classification task, alias."""

    pass


class GraphClassificationTask(ClassificationTask):
    """Graph classification task, alias."""

    pass


class NodeRegressionTask(RegressionTask):
    """Node regression task, alias."""

    pass


class GraphRegressionTask(RegressionTask):
    """Graph regression task, alias."""

    pass


class LinkPredictionTask(GLBTask):
    """Link prediction task."""

    def __init__(self, task_dict, pwd):
        """Link/Edge prediction."""
        self.target = "Edge/_Edge"
        self.valid_neg = task_dict.get("valid_neg", None)
        self.test_neg = task_dict.get("test_neg", None)
        self.sample_runtime = self.valid_neg is not None
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
    elif task_dict["type"] == "NodeRegression":
        return NodeRegressionTask(task_dict, pwd)
    elif task_dict["type"] == "GraphRegression":
        return GraphRegressionTask(task_dict, pwd)
    else:
        raise NotImplementedError(f"Unrecognized task: {task_dict['type']}"
                                  f"Supported tasks: {SUPPORTED_TASK_TYPES}")
