"""Automated test for metadata.json in datasets/."""
import pytest
import os
import json
from utils import find_datasets, check_if_task_json, \
    SUPPORTED_TASK_REQUIRED_KEYS_HASH, find_datasets_abs_path


def check_essential_keys_task_json(dic):
    """Check if task json has all essential keys."""
    missing_keys = []
    if "type" not in dic:
        missing_keys.append("type")
    else:
        task_type = dic.get("type")
        for req_keywords in SUPPORTED_TASK_REQUIRED_KEYS_HASH[task_type]:
            if req_keywords == "feature":
                if req_keywords not in dic:
                    missing_keys.append(req_keywords)
            else:
                if dic.get(req_keywords, None) is None:
                    missing_keys.append(req_keywords)
    return missing_keys


@pytest.mark.parametrize("dataset_name", find_datasets())
def test_task_json_content(dataset_name):
    """Check if task json meets requirements."""
    file_list = []
    directory = find_datasets_abs_path(dataset_name)

    for root, _, file in os.walk(directory):
        if isinstance(file, str):
            file.append(os.path.join(root, file))
            file_list.append(file)
        else:
            for f in file:
                file_list.append(os.path.join(root, f))
    for file in file_list:
        if check_if_task_json(file):
            with open(file, encoding="utf8") as json_file:
                data = json.load(json_file)
                missing_keys = check_essential_keys_task_json(data)
                if len(missing_keys) != 0:
                    print(file + " misses following keys")
                    print(missing_keys)
                assert len(missing_keys) == 0
