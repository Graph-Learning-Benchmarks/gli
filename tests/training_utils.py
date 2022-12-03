"""Functions used in test_training."""
import os
import fnmatch
import json


def get_cfg(dataset):
    """Return fixed dict to test_training."""
    args = {
        "model": "GCN",
        "dataset": dataset,
        "task": "NodeClassification",
        "gpu": -1
    }

    model_cfg = {
        "num_layers": 2,
        "num_hidden": 8,
        "dropout": .6
    }

    train_cfg = {
        "loss_fun": "cross_entropy",
        "dataset": {
            "self_loop": True,
            "to_dense": True
        },
        "optim": {
            "lr": .005,
            "weight_decay": 0.0005
        },
        "num_trials": 1,
        "max_epoch": 3
    }
    return args, model_cfg, train_cfg


def check_multiple_split_v2(dataset):
    """Check whether the dataset has multiple splits."""
    print()
    dataset_directory = os.getcwd() \
        + "/datasets/" + dataset
    for file in os.listdir(dataset_directory):
        if fnmatch.fnmatch(file, "task*.json"):
            with open(dataset_directory + "/" + file,  encoding="utf-8") as f:
                task_dict = json.load(f)
                if "num_splits" in task_dict and task_dict["num_splits"] > 1:
                    return 1
                else:
                    return 0


def check_dataset_task(dataset, target_task):
    """Check whether the dataset support target_task."""
    directory = os.getcwd() + "/datasets/" + dataset
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, "task*.json"):
            with open(directory + "/" + file,  encoding="utf-8") as f:
                task_dict = json.load(f)
                if task_dict["type"] == target_task:
                    return True
    return False


def get_label_number(labels):
    """Return the label number of dataset."""
    if len(labels.shape) > 1:
        return labels.shape[1]
    else:
        return 1
