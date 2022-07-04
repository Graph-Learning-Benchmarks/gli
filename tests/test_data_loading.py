"""Automated test for data files in examples/."""
import pytest
import os
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

    try:
        _ = glb.dataloading.get_glb_graph(dataset)
    except (AssertionError,
            AttributeError,
            ModuleNotFoundError,
            IndexError,
            ValueError) as e:
        print(e, dataset, "graph loading failed")
        assert False

    task_type = "task"
    try:
        _ = glb.dataloading.get_glb_task(dataset, task_type)
    except (AssertionError,
            AttributeError,
            ModuleNotFoundError,
            IndexError,
            ValueError) as e:
        print(e, dataset, task_type, "task loading failed")
        assert False

    try:
        glb.dataloading.get_glb_dataset(dataset, task_type)
    except (AssertionError,
            AttributeError,
            ModuleNotFoundError,
            IndexError,
            ValueError) as e:
        print(e, dataset, "combine graph and task loading failed")
        assert False