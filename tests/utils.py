"""Common functions for testing."""
import os
import re
import yaml

SUPPORTED_TASK_REQUIRED_KEYS_HASH = {
    "NodeClassification": [
        "description",
        "type",
        "feature",
        "target",
        "num_classes",
    ],
    "NodeRegression": [
        "description",
        "type",
        "feature",
        "target",
    ],
    "GraphClassification": [
        "description",
        "type",
        "feature",
        "target",
    ],
    "LinkPrediction": [
        "description",
        "type",
        "feature"
    ],
    "TimeDependentLinkPrediction": [
        "description",
        "type",
        "feature",
        "time",
        "train_time_window",
        "val_time_window",
        "test_time_window",
    ],
    "KGRelationPrediction": [
        "description",
        "type",
        "feature",
        "target",
        "num_relations",
        "train_triplet_set",
        "val_triplet_set",
        "test_triplet_set",
    ],
    "KGEntityPrediction": [
        "description",
        "type",
        "feature",
        "train_triplet_set",
        "val_triplet_set",
        "test_triplet_set",
    ],
    "GraphRegression": []
}


def dataset_dir_check(path_to_parent):
    """Check if a directory has metadata.json.

    If has then it is a dataset dir.
    """
    for fname in os.listdir(path_to_parent):
        if fname == "metadata.json":
            return True
    return False


def find_datasets_abs_path(dataset):
    """Find dataset absolute path."""
    return os.getcwd() + "/datasets/" + dataset


def find_datasets():
    """Get the datasets with changed files or find all datasets."""
    walk_dir = os.getcwd() + "/datasets"

    print("walk_dir = " + walk_dir)

    print("walk_dir (absolute) = " + os.path.abspath(walk_dir))

    if os.path.exists("temp/changed_datasets"):
        with open("temp/changed_datasets", encoding="utf-8") as f:
            dataset_dir_list = f.read().split()
        return dataset_dir_list

    dataset_dir_list = []
    for root, subdirs, _ in os.walk(walk_dir):
        for subdir in subdirs:
            if dataset_dir_check(root + "/" + subdir):
                dataset_dir_list.append(subdir)
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
    """Check if it is urls.json and correctly named."""
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


def check_if_license(file):
    """Check if it is LICENSE and correctly named."""
    if file == "LICENSE":
        return True
    return False


def check_if_converting_code(file, dataset):
    """Check if it is .ipynb or .py file and correctly named."""
    if file in (f"{dataset}.ipynb", file == f"{dataset}.py"):
        return True
    return False


def _dict_depth(d):
    """Return the depth of a dictionary."""
    if isinstance(d, dict):
        return 1 + (max(map(_dict_depth, d.values())) if d else 0)
    return 0


def _is_hetero_graph(data):
    """Return true if the gli data contains heterogeneous graph."""
    depth = _dict_depth(data)
    # Heterogeneous graph has one more depth than a homogeneous one.
    if depth == 5:
        return True
    elif depth == 4:
        return False
    else:
        raise RuntimeError("metadata.json has wrong structure.")


def load_config_file(path):
    """Load yaml files."""
    with open(path, "r", encoding="utf-8") as stream:
        try:
            parsed_yaml = yaml.full_load(stream)
            print(parsed_yaml)
            return parsed_yaml
        except yaml.YAMLError as exc:
            print(exc)


def _is_sparse_npz(filename):
    """Check if the npz file is storing a scipy sparse array."""
    return filename.endswith(".sparse.npz")
