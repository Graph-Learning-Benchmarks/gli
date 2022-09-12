# Graph Learning Indexer (GLI)

[![Pycodestyle](https://github.com/Graph-Learning-Benchmarks/gli/actions/workflows/pycodestyle.yml/badge.svg)](https://github.com/Graph-Learning-Benchmarks/gli/actions/workflows/pycodestyle.yml)
[![Pydocstyle](https://github.com/Graph-Learning-Benchmarks/gli/actions/workflows/pydocstyle.yml/badge.svg)](https://github.com/Graph-Learning-Benchmarks/gli/actions/workflows/pydocstyle.yml)
[![Pylint](https://github.com/Graph-Learning-Benchmarks/gli/actions/workflows/pylint.yml/badge.svg)](https://github.com/Graph-Learning-Benchmarks/gli/actions/workflows/pylint.yml)
[![Pytest](https://github.com/Graph-Learning-Benchmarks/gli/actions/workflows/pytest.yml/badge.svg)](https://github.com/Graph-Learning-Benchmarks/gli/actions/workflows/pytest.yml)

GLI is an easy-to-use graph learning platform with unique features that can better serve the dataset contributors, in comparison to existing graph learning libraries. It aims to ease and incentivize the creation and curation of datasets.

## Highlighted Features

### Standard Data Format

GLI defines a standard data format that has efficient storage and access for graphs. It unifies the storage for graphs of different scales and heterogeneity and is thus flexible to accommodate various graph-structured data.

### Explicit Separation of Data Storage and Task Configuration

GLI makes an explicit separation between the data storage and the task configuration for graph learning. i.e., Multiple tasks can be performed on the same dataset, or the same task can be performed on different datasets. The separation between graphs and tasks further allows users to use general datasets bound to every type of task that can be applied to every graph dataset.

<!-- TODO: Add more highlighted features. -->

## Get Started

### Installation

Currently, we support installation from the source.

```bash
git clone https://github.com/Graph-Learning-Benchmarks/gli.git
cd gli
pip install -e .
```

To test the installation, run the following command:

```bash
python example.py --graph cora --task NodeClassification
```

The output should be like this:

```
> Graph(s) loading takes 0.0196 seconds and uses 0.9788 MB.
> Task loading takes 0.0016 seconds and uses 0.1218 MB.
> Combining(s) graph and task takes 0.0037 seconds and uses 0.0116 MB.
Dataset("CORA dataset. NodeClassification", num_graphs=1, save_path=/Users/jimmy/.dgl/CORA dataset. NodeClassification)**
```

### Data Loading API

To load a dataset from the remote data repository, simply use the `get_gli_dataset()` function:

```python
>>> import gli
>>> dataset = gli.get_gli_dataset(dataset="cora", task="NodeClassification", device="cpu")
>>> dataset
Dataset("CORA dataset. NodeClassification", num_graphs=1, save_path=/Users/jimmy/.dgl/CORA dataset. NodeClassification)
```

Alternatively, one can also get a single graph or a list of graphs rather than a wrapped dataset by `get_gli_graph()`. Furthermore, GLI provides abstractions for various tasks (`GLITask`) and provides a function `get_gli_task()` to return a task instance. Combine these two instances to get a wrapped dataset that is identical to the previous case.

```python
>>> import gli
>>> g = gli.get_gli_graph(dataset="cora", device="cpu", verbose=False)
>>> g
Graph(num_nodes=2708, num_edges=10556,
      ndata_schemes={'NodeFeature': Scheme(shape=(1433,), dtype=torch.float32), 'NodeLabel': Scheme(shape=(), dtype=torch.int64)}
      edata_schemes={})
>>> task = gli.get_gli_task(dataset="cora", task="NodeClassification", verbose=False)
>>> task
<gli.task.NodeClassificationTask object at 0x100eff640>
>>> dataset = gli.combine_graph_and_task(g, task)
>>> dataset
Dataset("CORA dataset. NodeClassification", num_graphs=1, save_path=/Users/jimmy/.dgl/CORA dataset. NodeClassification)
```

The returned dataset is inherited from `DGLDataset`. Therefore, it can be incorporated into DGL's infrastructure seamlessly:

```python
>>> type(dataset)
<class 'gli.dataset.NodeClassificationDataset'>
>>> isinstance(dataset, dgl.data.DGLDataset)
True
```

## Contributing

### New Dataset, Feature Request, Bug Fix, or Better Documentation.

All kinds of improvement are welcomed! Please refer to our [contribution guide](CONTRIBUTING.md) for details.
