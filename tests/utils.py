"""Common functions for testing."""
import os

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

    example_dir_list = []
    for root, subdirs, _ in os.walk(walk_dir):
        for subdir in subdirs:
            if dataset_dir_check(root + "/" + subdir):
                example_dir_list.append(root + "/" + subdir)
    return example_dir_list
