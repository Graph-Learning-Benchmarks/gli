"""Automated test for json files in examples/."""
import pytest
import os
import re
import json


def example_dir_check(path_to_parent):
    """Check if a directory has metadata.json.

    If has then it is a example dir.
    """
    for fname in os.listdir(path_to_parent):
        if fname == "metadata.json":
            return True
    return False


def find_examples_dir():
    """Recursively find example directories which have no sub directories."""
    walk_dir = os.getcwd() + "/examples"

    print("walk_dir = " + walk_dir)

    print("walk_dir (absolute) = " + os.path.abspath(walk_dir))

    example_dir_list = []
    for root, subdirs, _ in os.walk(walk_dir):
        for subdir in subdirs:
            if example_dir_check(root + "/" + subdir):
                example_dir_list.append(root + "/" + subdir)
    return example_dir_list


def check_if_task_json(file):
    """Check if it is task.json file."""
    task_exp = re.compile(r"^task.*")
    json_exp = re.compile(r"\B.*\.json$")
    if task_exp.search(file) and json_exp.search(file):
        return True
    return False


def check_if_metadata_json(file):
    """Check if the file is correctly named."""
    if file == "metadata.json":
        return True
    return False


def check_file_name(files):
    """Check if correctly named task json and metadata json exist."""
    task_json_flag = False
    metadata_jason_flag = False

    for file in files:
        if check_if_task_json(file):
            task_json_flag = True
        if check_if_metadata_json(file):
            metadata_jason_flag = True

    if not task_json_flag:
        return False, "needs task json"

    if not metadata_jason_flag:
        return False, "needs metadata json"

    return True, "essential json files included"


@pytest.mark.parametrize("directory", find_examples_dir())
def test_if_has_essential_json(directory):
    """Check for essential json files.

    Recursively check if task.json file(s) and
    metadata.json file exist in all examples.
    """
    violations = None
    result = check_file_name(os.listdir(directory))
    if result[0] is True:
        pass
    else:
        violations = (directory, result[1])

    if violations is not None:
        print("example at " + violations[0] + " " + violations[1],
              "or json file(s) is not correctly named.\n")
    assert violations is None


def check_essential_keys_task_json(dic):
    """Check if task json has all essential keys."""
    missing_keys = []
    if dic.get("description", None) is None:
        missing_keys.append("description")

    if dic.get("type", None) is None:
        missing_keys.append("type")

    for temp_set in [("train_set", "train_time_window"),
                     ("val_set", "valid_time_window"),
                     ("test_set", "test_time_window")]:
        if dic.get(temp_set[0], None) is None and \
                dic.get(temp_set[1], None) is None:
            missing_keys.append(temp_set[0])
        else:
            if dic.get(temp_set[0], None) is None:
                pass
            else:
                for temp_key in ["file", "key"]:
                    if dic[temp_set[0]].get(temp_key, None) is None:
                        missing_keys.append(temp_set[0] + ":" + temp_key)
    return missing_keys


@pytest.mark.parametrize("directory", find_examples_dir())
def test_task_json_content(directory):
    """Check if task json meets requirements."""
    file_list = os.listdir(directory)
    for file in file_list:
        if check_if_task_json(file):
            with open(directory + "/" + file, encoding="utf8") as json_file:
                data = json.load(json_file)
                missing_keys = check_essential_keys_task_json(data)
                if len(missing_keys) != 0:
                    print(directory + "/" + file + " misses following keys")
                    print(missing_keys)
                assert len(missing_keys) == 0


def check_essential_keys_metadata_json(dic):
    """Check if task json has all essential keys."""
    missing_keys = []
    if dic.get("description", None) is None:
        missing_keys.append("description")

    if dic.get("data", None) is None:
        missing_keys.append("data")
    else:
        for first_key in ["Node", "Edge", "Graph"]:
            if dic["data"].get(first_key, None) is None:
                missing_keys.append("data: " + first_key)

    return missing_keys


@pytest.mark.parametrize("directory", find_examples_dir())
def test_metadata_json_content(directory):
    """Check if metadata json meets requirements."""
    file_list = os.listdir(directory)
    for file in file_list:
        if check_if_metadata_json(file):
            with open(directory + "/" + file, encoding="utf8") as json_file:
                data = json.load(json_file)
                missing_keys = check_essential_keys_metadata_json(data)
                if len(missing_keys) != 0:
                    print(directory + "/" + file + " misses following keys")
                    print(missing_keys)
                assert len(missing_keys) == 0
