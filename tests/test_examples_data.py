"""Automated test for data files in examples/."""
import pytest
import os
import re
import json
import sys


def example_data_check(path_to_parent):
    """Check if a directory has metadata.json and npz files.

    If has then it is a example dir.
    """
    metadata_flag = False
    data_flag = False

    npz_exp = re.compile(r"\B.*\.npz$")

    for fname in os.listdir(path_to_parent):
        if fname == "metadata.json":
            metadata_flag = True
        if npz_exp.search(fname):
            data_flag = True
    if data_flag and metadata_flag:
        return True
    return False


def find_examples_dir():
    """Recursively find example directories which have data files."""
    walk_dir = os.getcwd() + "/examples"

    print("walk_dir = " + walk_dir)

    print("walk_dir (absolute) = " + os.path.abspath(walk_dir))

    example_dir_list = []
    for root, subdirs, _ in os.walk(walk_dir):
        print(root)
        for subdir in subdirs:
            print(root + "/" + subdir)
            if example_data_check(root + "/" + subdir):
                example_dir_list.append(root + "/" + subdir)
    return example_dir_list


@pytest.mark.parametrize("directory", find_examples_dir())
def graph_loading_test(directory):
    """Test if glb.graph.read_glb_graph can be applied successfully"""
    sys.path.append(sys.path[0] + "/..")
    print(sys.path)
    import glb
    metadata_path = directory + "/metadata.json"
    try:
        g = glb.graph.read_glb_graph(metadata_path=metadata_path)
    except Exception as e:
        print(e, metadata_path, "graph loading failed")
        assert False

    task_path = directory + "/task.json"
    try:
        task = glb.task.read_glb_task(task_path=task_path)
    except Exception as e:
        print(e, task_path, "task loading failed")
        assert False

    try:
        glb.dataloading.combine_graph_and_task(g, task)
    except Exception as e:
        print(e, directory, "combine graph and task loading failed")
        assert False