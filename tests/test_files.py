"""Automated test for required files in datasets/."""
import pytest
import os
from utils import \
    find_datasets, check_if_metadata_json, \
    check_if_urls_json, check_if_task_json, check_if_readme, \
    find_datasets_abs_path


def check_file_name(files):
    """Check if essential files exist.

    Correctly named task json, metadata json urls json, and README exist.
    """
    task_json_flag = False
    metadata_json_flag = False
    urls_jason_flag = False
    readme_flag = False

    for file in files:
        if check_if_task_json(file):
            task_json_flag = True
        if check_if_metadata_json(file):
            metadata_json_flag = True
        if check_if_urls_json(file):
            urls_jason_flag = True
        if check_if_readme(file):
            readme_flag = True

    if not task_json_flag:
        return False, "needs task json"

    if not metadata_json_flag:
        return False, "needs metadata json"

    if not urls_jason_flag:
        return False, "needs urls json"

    if not readme_flag:
        return False, "needs README"

    return True, "essential files included"


@pytest.mark.parametrize("dataset_name", find_datasets())
def test_if_has_essential_files(dataset_name):
    """Check for essential json files.

    Recursively check if task.json file(s) and
    metadata.json file exist in all datasets.
    """
    violations = None
    directory = find_datasets_abs_path(dataset_name)
    result = check_file_name(os.listdir(directory))

    if result[0] is True:
        pass
    else:
        violations = (directory, result[1])

    if violations is not None:
        print("dataset at " + violations[0] + " " + violations[1],
              "or json file(s) is not correctly named.\n")
    assert violations is None
