"""Task for GLB."""
import json
import os
from typing import List

from glb.utils import load_data

SUPPORT_TASK_TYPES = ["NodeClassification", "GraphClassification"]


class GLBTask:
    """GLB task base class."""
    def __init__(self, task_dict, pwd):
        """Initialize GLBTask."""
        self.pwd = pwd
        self.type = task_dict["type"]
        self.description = task_dict["description"]
        self.features: List[str] = task_dict["feature"]
        self.target: str = task_dict["target"]
        self.split = {"train_set": None, "val_set": None, "test_set": None}

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


class ClassificationTask(GLBTask):
    """Classification task."""
    def __init__(self, task_dict, pwd):
        super().__init__(task_dict, pwd)
        self.num_classes = task_dict["num_classes"]


class NodeClassificationTask(ClassificationTask):
    """Node classification task, alias."""
    pass


class GraphClassificationTask(ClassificationTask):
    """Graph classification task, alias."""
    pass


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
    else:
        raise NotImplementedError(f"Unrecognized task: {task_dict['type']}"
                                  f"Supported tasks: {SUPPORT_TASK_TYPES}")
