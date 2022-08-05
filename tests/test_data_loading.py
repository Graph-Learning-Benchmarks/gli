"""Automated test for data files in examples/."""
import os
import fnmatch
import json
import pytest
import glb
from glb.task import SUPPORTED_TASK_TYPES
from utils import find_datasets_dir


@pytest.mark.parametrize("dataset_name", find_datasets_dir())
def test_data_loading(dataset_name):
    """Test data loading for a given dataset.

    Test if get_glb_graph, get_glb_task, and get_glb_dataset
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
            print(os.path.splitext(file)[0])
            task_list.append(os.path.splitext(file)[0])
    try:
        _ = glb.dataloading.get_glb_graph(dataset)
    except (AssertionError,
            AttributeError,
            ModuleNotFoundError,
            IndexError,
            ValueError) as e:
        print(e, dataset, "graph loading failed")
        assert False

    for task in task_list:
        try:
            _ = glb.dataloading.get_glb_task(dataset, task)
        except (AssertionError,
                AttributeError,
                ModuleNotFoundError,
                IndexError,
                ValueError) as e:
            print(e, dataset, task, "task loading failed")
            assert False

        try:
            glb.dataloading.get_glb_dataset(dataset, task)
        except (AssertionError,
                AttributeError,
                ModuleNotFoundError,
                IndexError,
                ValueError) as e:
            print(e, dataset, "combine graph and task loading failed")
            assert False
