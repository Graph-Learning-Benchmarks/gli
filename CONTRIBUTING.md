# Contributing to GLI

If you are interested in contributing to GLI, your contributions will likely fall into one of the following three categories:

1. You want to contribute a new dataset/task.
2. You want to implement a new feature.
3. You want to fix a bug.

## Developing GLI

To develop GLI on your machine, here are some tips:

1. Clone a copy of GLI from source:

   ```bash
   git clone https://github.com/Graph-Learning-Benchmarks/gli.git
   cd gli
   ```

2. If you already cloned GLI from source, update it:

   ```
   git pull
   ```

3. Install GLI in editable mode:

   ```
   pip install -e ".[test,full]"
   ```

   This mode will symlink the Python files from the current local source tree into the Python install. Hence, if you modify a Python file, you do not need to reinstall GLI again.

4. Run an example:

   ```
   python3 example.py
   ```

   This script will load the `NodeClassification` task on `cora` dataset.

5. Ensure your installation is correct by running the entire test suite with

   ```
   pytest
   ```

## Contributing A New Dataset

Here is a checklist for a new dataset. Please [open a pull request](https://github.com/Graph-Learning-Benchmarks/gli/pulls?q=is%3Apr+is%3Aopen) that contains a directory in following format:

```
datasets/<name>
├── <name>.ipynb/<name>.py
├── README.md
├── metadata.json
├── task_<task_type>.json
├── ...  # There might be multiple task configurations.
└── urls.json
```

where `<name>` is the dataset name and `<task_type>` is one of the given tasks defined in [GLI Task Format](FORMAT.md#gli-task-format).

- `<name>.ipynb/<name>.py`: A Jupyter Notebook or Python script that converts the original dataset into GLI format.
- `README.md`: A document that contains the necessary information about the dataset and task(s), including description, citation(s), available task(s), and extra required packages for `<name>.ipynb/<name>.py`.
- `metadata.json`: A json configuration file that stores the metadata of the graph dataset. See [GLI Data Format](FORMAT.md#gli-data-format).
- `task_<task_type>.json`：A task configuration file that stores an available task on the given dataset. See [GLI Task Format](FORMAT.md#gli-task-format). Contributors can define multiple tasks on the same dataset. If the task type is the same, use `task_<task_type>_<id>.json` to distinguish between same tasks, where `<id>` should be replaced by 1, 2, etc.
- `urls.json`: A url configuration file that stores the downloading urls of the uploaded files. See [Uploading GLI Data Files](#uploading-gli-data-files).

### Uploading GLI Data Files

Please upload the npz or npy files referred in `metadata.json` or `task_<task_type>.json` to a cloud storage platform (e.g., Dropbox or Google Drive), and include the public download links in `urls.json`.

### Data Removal Policy

The original contributor(s) of a dataset may request a removal of the dataset. Project maintainers will notify all the subsequent contributors to that dataset, and remove the dataset from the main branch and the cloud storage platform within 30 days, if there is no reasonable objection from the subsequent contributors. 

**Warning:** It may not be possible to remove the datasets completely once contributed to GLI. The metadata information will remain in the commit history, and there might be other distributions on the internet.


## Reporting Bugs

Please feel free to [report a bug through Issues](https://github.com/Graph-Learning-Benchmarks/gli/issues/new?assignees=&labels=bug&template=bug_report.md&title=%5BBUG%5D) and/or [open a pull request to implement it](https://github.com/Graph-Learning-Benchmarks/gli/pulls?q=is%3Apr+is%3Aopen). Please provide a clear and concise description of what the bug was. If you are unsure about if this is a bug at all or how to fix, post about it in an issue.

## Implementing New Features

Please feel free to [request a new feature through Issues](https://github.com/Graph-Learning-Benchmarks/gli/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=%5BFEATURE+REQUEST%5D) and/or [open a pull request to implement it](https://github.com/Graph-Learning-Benchmarks/gli/pulls?q=is%3Apr+is%3Aopen). In general, we accept any features as long as they fit the scope of this package. If you are unsure about this or need help on the design/implementation of your feature, post about it in an issue.
