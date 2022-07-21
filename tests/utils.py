"""Common functions for testing."""
import os
import re

CURRENT_SUPPORT_TASKS = [
    "NodeClassification",
    "GraphClassification",
    "TimeDependentLinkPrediction",
    "EntityLinkPrediction",
    "RelationLinkPrediction",
]


def dataset_dir_check(path_to_parent):
    """Check if a directory has metadata.json.

    If has then it is a example dir.
    """
    for fname in os.listdir(path_to_parent):
        if fname == "metadata.json":
            return True
    return False


def find_datasets_dir():
    """Recursively find dataset directories which have metadata json."""
    walk_dir = os.getcwd() + "/datasets"

    print("walk_dir = " + walk_dir)

    print("walk_dir (absolute) = " + os.path.abspath(walk_dir))

    dataset_dir_list = []
    for root, subdirs, _ in os.walk(walk_dir):
        for subdir in subdirs:
            if dataset_dir_check(root + "/" + subdir):
                dataset_dir_list.append(root + "/" + subdir)
    return dataset_dir_list


def check_if_task_json(file):
    """Check if it is task.json file and correctly named."""
    task_exp = re.compile(r".*task.*")
    json_exp = re.compile(r"\B.*\.json$")
    if task_exp.search(file) and json_exp.search(file):
        return True
    return False


def check_if_metadata_json(file):
    """Check if it is metadata.json and correctly named."""
    if file == "metadata.json":
        return True
    return False


def check_if_urls_json(file):
    """Check if it is metadata.json and correctly named."""
    urls_exp = re.compile(r".*urls.*")
    json_exp = re.compile(r"\B.*\.json$")
    if urls_exp.search(file) and json_exp.search(file):
        return True
    return False


def check_if_readme(file):
    """Check if it is README.md and correctly named."""
    if file == "README.md":
        return True
    return False


def _dict_depth(d):
    """Return the depth of a dictionary."""
    if isinstance(d, dict):
        return 1 + (max(map(_dict_depth, d.values())) if d else 0)
    return 0


def _is_hetero_graph(data):
    """Return true if the glb data contains heterogeneous graph."""
    depth = _dict_depth(data)
    # Heterogeneous graph has one more depth than a homogeneous one.
    if depth == 5:
        return True
    elif depth == 4:
        return False
    else:
        raise RuntimeError("metadata.json has wrong structure.")
