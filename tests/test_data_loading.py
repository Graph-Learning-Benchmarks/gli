"""Automated test for data files in examples/."""
import os
import fnmatch
import json
import pytest
import gli
from gli.task import SUPPORTED_TASK_TYPES
from utils import find_datasets


@pytest.mark.parametrize("dataset_name", find_datasets())
def test_data_loading(dataset_name):
    """Test data loading for a given dataset.

    Test if get_gli_graph, get_gli_task, and get_gli_dataset
    can be applied successfully.
    """
    # temporary skipping all large datasets
    dataset = dataset_name
    large_dataset_to_skip = ["wiki", "ogbg-code2"]
    if dataset in large_dataset_to_skip:
        return

    directory = os.getcwd() + "/datasets/" + dataset
    task_list = []
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, "task*.json"):
            with open(directory + "/" + file,  encoding="utf-8") as f:
                task_dict = json.load(f)
                if task_dict["type"] not in SUPPORTED_TASK_TYPES:
                    f.close()
                    return
            task_list.append(task_dict["type"])
    try:
        _ = gli.dataloading.get_gli_graph(dataset)
    except (AssertionError,
            AttributeError,
            ModuleNotFoundError,
            IndexError,
            ValueError) as e:
        print(e, dataset, "graph loading failed")
        assert False

    for task in task_list:
        try:
            _ = gli.dataloading.get_gli_task(dataset, task)
        except (AssertionError,
                AttributeError,
                ModuleNotFoundError,
                IndexError,
                ValueError) as e:
            print(e, dataset, task, "task loading failed")
            assert False

        try:
            gli.dataloading.get_gli_dataset(dataset, task)
        except (AssertionError,
                AttributeError,
                ModuleNotFoundError,
                IndexError,
                ValueError) as e:
            print(e, dataset, "combine graph and task loading failed")
            assert False
