"""
``gli.task`` module.

The ``gli.task`` module contains task classes allowed by GLI and utilities for
loading them during runtime. Directly using the classes in this module is still
experimental. Users are encouraged to use the :func:`gli.task.read_gli_task`
to load tasks from files instead.

See details of GLI file-based task format in :ref:`format`.
"""

import json
import math
import os
import random
from typing import List

import torch

from gli.utils import load_data

SUPPORTED_TASK_TYPES = [
    "NodeClassification", "NodeRegression", "GraphClassification",
    "GraphRegression", "TimeDependentLinkPrediction", "LinkPrediction",
    "KGEntityPrediction", "KGRelationPrediction"
]


class GLITask:
    """The basic GLI task class for creating graph learning tasks.

    This class contains the necessary attributes and methods for all GLI tasks.

    :param task_dict: A dictionary containing the task information.
    :type task_dict: dict
    :param pwd: The path to the directory containing the task files.
    :type pwd: str
    :param device: The device to load the task data to.
    :type device: str

    Notes
    =====

    The ``task_dict`` should at least contain the following keys:

    * ``type``: The type of the task.
    * ``description``: A description of the task.
    * ``feature``: A list of features to use for the task.
    * ``target``: The target to use for the task.

    The :class:`gli.task.GLITask` class also contains split information. There
    are two split methods supported by GLI tasks: random split and predefined
    split.

    If the random split is used, the ``task_dict`` should contain the following
    keys:

    * ``train_ratio``: The ratio of training samples.
    * ``val_ratio``: The ratio of validation samples.
    * ``test_ratio``: The ratio of test samples.
    * ``num_samples``: The number of samples used in random sampling.
    * ``seed``: The random seed to use for random sampling. (optional)

    Otherwise, in the predefined split method, the ``task_dict`` should contain
    the following keys:

    * ``train_set``: The path to the file containing the training set.
    * ``val_set``: The path to the file containing the validation set.
    * ``test_set``: The path to the file containing the test set.

    Other optional keys include:

    * ``num_splits``: The number of splits to use for the task. (optional, 1 by
      default)

    Warning
    -------

    Instantiating a :class:`gli.task.GLITask` object directly is still
    experimental. Instead, you should use the :func:`gli.task.read_gli_task` or
    :func:`gli.dataloading.get_gli_task` to load tasks from files.

    """

    def __init__(self, task_dict, pwd, device="cpu"):
        """Initialize GLITask.

        This method will infer whether to use random split or predefined split
        and then load the split information into :attr:`.GLITask.splits`.
        """
        self.pwd = pwd
        self.type = task_dict["type"]
        self.description = task_dict["description"]
        self.features: List[str] = task_dict["feature"]
        self.target: str = None
        self.num_splits = 1
        self.random_split = False
        self.split = {"train_set": None, "val_set": None, "test_set": None}
        self.device = device

        if not any([
                "train_set" in task_dict, "train_time_window" in task_dict,
                "train_triplet_set" in task_dict
        ]):
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
                        load_data(path, this_fold_key, self.device))
                    # can be a list of mask tensors or index tensors
            else:
                self.split[dataset_] = load_data(path, key, self.device)
                # can be a mask tensor or an index tensor


class ClassificationTask(GLITask):
    """Classification task."""

    def __init__(self, task_dict, pwd, device="cpu"):
        """Initialize num_classes."""
        super().__init__(task_dict, pwd, device)
        self.num_classes = task_dict["num_classes"]
        self.target = task_dict["target"]

    def _load(self, task_dict):
        self._load_split(task_dict)


class RegressionTask(GLITask):
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


class LinkPredictionTask(GLITask):
    """Link prediction task."""

    def __init__(self, task_dict, pwd):
        """Link/Edge prediction."""
        self.target = "Edge/_Edge"
        self.val_neg = task_dict.get("val_neg", None)
        self.test_neg = task_dict.get("test_neg", None)
        self.sample_runtime = self.val_neg is None
        super().__init__(task_dict, pwd)

    def _load(self, task_dict):
        self._load_split(task_dict)


class KGEntityPredictionTask(GLITask):
    """Knowledge graph entity prediction task."""

    def __init__(self, task_dict, pwd, device="cpu"):
        """Initialize KGEntityPredictionTask."""
        # REVIEW - only supports runtime sampling for now
        self.sample_runtime = True
        self.num_relations = task_dict["num_relations"]
        # REVIEW - Making predict_tail optional to be compatible
        # with existing datasets. Should be removed when datasets
        # are updated with the new save_task helper function.
        self.predict_tail = task_dict.get(
            "predict_tail", True)
        super().__init__(task_dict, pwd, device)

    def _load(self, task_dict):
        self._load_split(task_dict)

    def _load_split(self, task_dict: dict):
        for k in ("train", "val", "test"):
            task_dict[f"{k}_set"] = task_dict.pop(f"{k}_triplet_set")
        super()._load_split(task_dict)


class KGRelationPredictionTask(GLITask):
    """Knowledge graph relation prediction task."""

    def __init__(self, task_dict, pwd, device="cpu"):
        """Rename num_relations to num_classes."""
        # REVIEW - only supports runtime sampling for now
        self.sample_runtime = True
        self.num_relations = task_dict["num_relations"]
        self.target = task_dict["target"]
        super().__init__(task_dict, pwd, device)
        self.target = task_dict["target"]

    def _load(self, task_dict):
        self._load_split(task_dict)

    def _load_split(self, task_dict: dict):
        for k in ("train", "val", "test"):
            task_dict[f"{k}_set"] = task_dict.pop(f"{k}_triplet_set")
        super()._load_split(task_dict)


class TimeDependentLinkPredictionTask(LinkPredictionTask):
    """Time dependent link prediction task."""

    def __init__(self, task_dict, pwd):
        """Time dependent link prediction task."""
        self.time = task_dict["time"]
        self.time_window = {
            "train_time_window": task_dict["train_time_window"],
            "val_time_window": task_dict["val_time_window"],
            "test_time_window": task_dict["test_time_window"]
        }
        super().__init__(task_dict, pwd)

    def _load(self, task_dict):
        for neg_idx in ["val_neg", "test_neg"]:
            if getattr(self, neg_idx, None):
                filename = task_dict[neg_idx]["file"]
                key = task_dict[neg_idx].get("key")
                path = os.path.join(self.pwd, filename)
                indices = load_data(path, key, self.device)
                setattr(self, neg_idx, indices)


def read_gli_task(task_path: str, verbose=True):
    """Read a local GLI task file and return a task object.

    :param task_path: Path to the task file.
    :type task_path: str
    :param verbose: Whether to print the task description, defaults to True.
    :type verbose: bool, optional
    :return: A task object.
    :rtype: :class:`gli.task.GLITask`

    Notes
    -----
    This function is used to read a GLI task file locally. It is not used to
    fetch a task configuration from a remote server. If you want to download
    any task configuration provided by GLI, use
    :func:`gli.dataloading.get_gli_task` instead.

    Additionally, this function is useful when you want to test loading a new
    task configuration file locally.
    """
    pwd = os.path.dirname(task_path)
    with open(task_path, "r", encoding="utf-8") as fptr:
        task_dict = json.load(fptr)
    if verbose:
        print(task_dict["description"])

    if task_dict["type"] in SUPPORTED_TASK_TYPES:
        # Call class constructors by Python eval() method
        # This method is discouraged by pylint but we have limited the type.
        # pylint: disable=W0123
        return eval(task_dict["type"] + "Task")(task_dict, pwd)
    else:
        raise NotImplementedError(f"Unrecognized task: {task_dict['type']}"
                                  f"Supported tasks: {SUPPORTED_TASK_TYPES}")
