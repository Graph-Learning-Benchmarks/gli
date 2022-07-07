"""Automated test for data files in examples/."""
import pytest
import os
import fnmatch
import json
import glb


def find_examples_dir():
    """Find example directories which have data files."""
    walk_dir = os.getcwd() + "/datasets"

    print("walk_dir = " + walk_dir)

    print("walk_dir (absolute) = " + os.path.abspath(walk_dir))

    example_dir_list = []
    for root, subdirs, _ in os.walk(walk_dir):
        print(root)
        for subdir in subdirs:
            print(root + "/" + subdir)
            example_dir_list.append(root + "/" + subdir)
    return example_dir_list


@pytest.mark.parametrize("directory", find_examples_dir())
def test_graph_loading(directory):
    """Test if glb.graph.get_glb_graph can be applied successfully."""
    dataset = os.path.basename(directory)
    task_list = []
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, 'task*.json'):
            f = open(directory + "/" + file)
            task_dict = json.load(f)
            if "Link" in task_dict["type"]:
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
