"""Automated test for data files in examples/."""
import os
import fnmatch
import json
import glb


def test_graph_loading(dataset):
    """Test if glb.graph.get_glb_graph can be applied successfully."""
    directory = os.getcwd() + "/" + dataset
    task_list = []
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, "task*.json"):
            with open(directory + "/" + file,  encoding="utf-8") as f:
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
