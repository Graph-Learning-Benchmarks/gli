"""Automated test for required files in datasets/."""
import pytest
import os
from utils import \
    find_datasets, check_if_metadata_json, \
    check_if_urls_json, check_if_task_json, check_if_readme, \
    check_if_license, check_if_converting_code, find_datasets_abs_path


def check_file_name(files, dataset_name):
    """Check if essential files exist.

    Correctly named task json, metadata json urls json, and README exist.
    """
    task_json_flag = False
    metadata_json_flag = False
    urls_jason_flag = False
    readme_flag = False
    license_flag = False
    coverting_code_flag = False

    for file in files:
        if check_if_task_json(file):
            task_json_flag = True
        if check_if_metadata_json(file):
            metadata_json_flag = True
        if check_if_urls_json(file):
            urls_jason_flag = True
        if check_if_readme(file):
            readme_flag = True
        if check_if_license(file):
            license_flag = True
        if check_if_converting_code(file, dataset_name):
            coverting_code_flag = True

    missing_file_messages_error = []
    missing_file_messages_warning = []
    if not task_json_flag:
        missing_file_messages_warning.append("missing task_*.json file(s)")

    if not metadata_json_flag:
        missing_file_messages_error.append("missing metadata.json")

    if not urls_jason_flag:
        missing_file_messages_warning.append("missing urls.json")

    if not readme_flag:
        missing_file_messages_error.append("missing README.md")

    if not license_flag:
        missing_file_messages_error.append("missing LICENSE")

    if not coverting_code_flag:
        missing_file_messages_error.append(
            "missing data conversion code (<dataset>.ipynb or <dataset>.py)")

    return missing_file_messages_error, missing_file_messages_warning


@pytest.mark.parametrize("dataset_name", find_datasets())
def test_if_has_essential_files(dataset_name):
    """Check for essential json files.

    Recursively check if task.json file(s) and
    metadata.json file exist in all datasets.
    """
    directory = find_datasets_abs_path(dataset_name)
    errors, warnings = check_file_name(os.listdir(directory), dataset_name)

    if len(errors) > 0:
        print(f"Required files are missing for the dataset at {directory}:")
        for err in errors:
            print(err)
        print(
            "Please check if the above file(s) are present and "
            "correctly named.")
    assert len(errors) == 0
    if len(warnings) > 0:
        print(f"Required files are missing for the dataset at {directory}:")
        for warn in warnings:
            print(warn)
        print(
            "Please check if the above file(s) are present and "
            "correctly named.")
