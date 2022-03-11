"""Task for GLB."""
import json
import os

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
