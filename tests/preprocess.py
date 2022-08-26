"""Preprocess before tests."""
import os
import shutil
import subprocess

DATAFILES_URL = "https://www.dropbox.com/s/ltt1q7i7bh3cu2h/datafiles.tar?dl=0"
NUM_TESTS_THRESHOLD = 15


def _prepare_data_files():
    if os.path.exists("temp/changed_datasets"):
        with open("temp/changed_datasets", encoding="utf-8") as f:
            dataset_dir_list = f.read().split()
        if len(dataset_dir_list) < NUM_TESTS_THRESHOLD:
            # do not download the combined data files if # of tests is small
            return
    out = "datafiles.tar"
    url = DATAFILES_URL
    subprocess.run(["wget", "-q", "-O", out, url], check=True)
    shutil.unpack_archive(out)
    os.remove(out)

    for dataset in os.listdir("datafiles/"):
        data_dir = os.path.join("datafiles/", dataset)
        dataset_dir = os.path.join("datasets/", dataset)
        if not os.path.isdir(data_dir):
            continue
        for data_file_name in os.listdir(data_dir):
            file_type = os.path.splitext(data_file_name)[-1]
            if file_type == ".npz":
                data_file_path = os.path.join(data_dir, data_file_name)
                dataset_file_path = os.path.join(dataset_dir, data_file_name)
                shutil.move(data_file_path, dataset_file_path)
    shutil.rmtree("datafiles/")


if __name__ == "__main__":
    _prepare_data_files()
