# Contributing to GLI

If you are interested in contributing to GLI, your contributions will likely fall into one of the following three categories:

1. You want to [contribute a new dataset/task](#contributing-a-new-dataset-or-task).
2. You want to [implement a new feature for GLI](#implementing-a-new-feature-for-gli).
3. You want to [fix a bug](#reporting-bugs).

We also provide some times on how to [start developing GLI on your local machine](#tips-on-developing-gli).

## Contributing A New Dataset or Task

If you want to contribute a new dataset or task, please take a look at the following three sections for information relevant to dataset contribution.

### Dataset Contribution Workflow

Detailed instructions on the dataset contribution workflow can be found in the [Dataset Submission Guideline
](https://github.com/Graph-Learning-Benchmarks/gli/wiki/Dataset-Submission-Guideline).

### Code of Ethics on Dataset Contribution

Ethical considerations are critical components of the quality of datasets. We have provided our code of ethics on dataset contribution in [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md#code-of-ethics-on-dataset-contribution).

### Data Removal Policy

The original contributor(s) of a dataset may request a removal of the dataset. Project maintainers will notify all the subsequent contributors to that dataset, and remove the dataset from the main branch and the cloud storage platform within 30 days, if there is no reasonable objection from the subsequent contributors.

**Warning:** It may not be possible to remove the datasets completely once contributed to GLI. The metadata information will remain in the commit history, and there might be other distributions on the internet.


## Implementing A New Feature for GLI

Please feel free to [request a new feature through Issues](https://github.com/Graph-Learning-Benchmarks/gli/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=%5BFEATURE+REQUEST%5D) and/or [open a pull request to implement it](https://github.com/Graph-Learning-Benchmarks/gli/pulls?q=is%3Apr+is%3Aopen). In general, we accept any features as long as they fit the scope of this package. If you are unsure about this or need help on the design/implementation of your feature, post about it in an issue.


## Reporting Bugs

Please feel free to [report a bug through Issues](https://github.com/Graph-Learning-Benchmarks/gli/issues/new?assignees=&labels=bug&template=bug_report.md&title=%5BBUG%5D) and/or [open a pull request to fix it](https://github.com/Graph-Learning-Benchmarks/gli/pulls?q=is%3Apr+is%3Aopen). Please provide a clear and concise description of what the bug was. If you are unsure about if this is a bug at all or how to fix, post about it in an issue.


## Tips on Developing GLI

To develop GLI on your local machine, here are some tips:

1. Clone a copy of GLI from source:

   ```bash
   git clone https://github.com/Graph-Learning-Benchmarks/gli.git
   cd gli
   ```

2. If you already cloned GLI from source, update it:

   ```bash
   git pull
   ```

3. Install GLI in editable mode:

   ```bash
   pip install -e ".[test,full]"
   ```

   This mode will symlink the Python files from the current local source tree into the Python install. Hence, if you modify a Python file, you do not need to reinstall GLI again.

4. Run an example:

   ```bash
   python3 example.py
   ```

   This script will load the `NodeClassification` task on `cora` dataset.

5. Ensure your installation is correct by running the entire test suite with

   ```bash
   make test
   ```