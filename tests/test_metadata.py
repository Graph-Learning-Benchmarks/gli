"""Automated test for metadata.json in datasets/."""
import pytest
import os
import json
from utils import find_datasets_dir, check_if_metadata_json, _is_hetero_graph, \
    find_datasets_abs_path


def check_essential_keys_metadata_json(dic):
    """Check if metadata json has all essential keys."""
    missing_keys = []
    if dic.get("description", None) is None:
        missing_keys.append("description")
    if dic.get("citation", None) is None:
        missing_keys.append("citation")
    if dic.get("is_heterogeneous", None) is None:
        missing_keys.append("is_heterogeneous")
    if dic.get("data", None) is None:
        missing_keys.append("data")
    else:
        for first_key in ["Node", "Edge", "Graph"]:
            if dic["data"].get(first_key, None) is None:
                missing_keys.append("data: " + first_key)

        if _is_hetero_graph(dic):
            missing_keys += \
                check_essential_keys_metadata_json_heterogeneous(dic)
            assert dic["is_heterogeneous"] is True
        else:
            missing_keys += check_essential_keys_metadata_json_homogeneous(dic)
            assert dic["is_heterogeneous"] is False
    return missing_keys


def check_essential_keys_metadata_json_homogeneous(dic):
    """Check if homogeneous meta json has all essential keys."""
    missing_keys = []
    for key in dic["data"]["Node"].keys():
        for sub_key in ["description",
                        "type",
                        "format",
                        "file",
                        "key",
                        ]:
            if dic["data"]["Node"][key].get(sub_key, None) is None:
                missing_keys.append("data: Node: " + key + ": " + sub_key)

    for sup_key in ["Edge", "Graph"]:
        for key in dic["data"][sup_key].keys():
            for sub_key in ["file",
                            "key",
                            ]:
                if dic["data"][sup_key][key].get(sub_key, None) is None:
                    missing_keys.append("data: " + sup_key + ": " +
                                        key + ": " + sub_key)

    return missing_keys


def check_essential_keys_metadata_json_heterogeneous(dic):
    """Check if heterogeneous meta json has all essential keys."""
    missing_keys = []
    for key in dic["data"]["Node"].keys():
        if dic["data"]["Node"][key].get("_ID", None) is None:
            missing_keys.append("data: Node: " + key + ": " + ": _ID")

        for sub_key in dic["data"]["Node"][key]:
            if dic["data"]["Node"][key]["_ID"].get("file", None) is None:
                missing_keys.append("data: Node: " + key + ": " +
                                    sub_key + ": _ID: file")

            if dic["data"]["Node"][key]["_ID"].get("key", None) is None:
                missing_keys.append("data: Node: " + key + ": " +
                                    sub_key + ": _ID: key")
                for sub_sub_key in ["description",
                                    "type",
                                    "format",
                                    "file",
                                    "key",
                                    ]:
                    if dic["data"]["Node"][key][sub_key].get(sub_sub_key,
                                                             None) is None:
                        missing_keys.append("data: Node: " +
                                            key + ": " + sub_key +
                                            ": " + sub_sub_key)

    for key in dic["data"]["Edge"].keys():
        for sub_key in ["_ID",
                        "_Edge",
                        ]:
            if dic["data"]["Edge"][key].get(sub_key, None) is None:
                missing_keys.append("data: Edge: " + key + ": " + sub_key)
            else:
                for sub_sub_key in ["file",
                                    "key",
                                    ]:
                    if dic["data"]["Edge"][key][sub_key].get(sub_sub_key,
                                                             None) is None:
                        missing_keys.append("data: Edge: " + key + ": " +
                                            sub_key + ": " + sub_sub_key)
    return missing_keys


@pytest.mark.parametrize("directory", find_datasets_dir())
def test_metadata_json_content(directory):
    """Check if metadata json meets requirements."""
    file_list = []
    directory = find_datasets_abs_path(directory)
    print(directory)
    for root, _, file in os.walk(directory):
        if isinstance(file, str):
            file.append(os.path.join(root, file))
            file_list.append(file)
        else:
            for f in file:
                file_list.append(f)
    for file in file_list:
        if check_if_metadata_json(file):
            with open(directory + "/" + file, encoding="utf8") as json_file:
                metadata = json.load(json_file)
                missing_keys = check_essential_keys_metadata_json(metadata)
                if len(missing_keys) != 0:
                    print(directory + "/" + file + " misses following keys")
                    print(missing_keys)
                assert len(missing_keys) == 0
