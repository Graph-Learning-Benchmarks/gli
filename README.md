<p align='center'>
  <img width='40%' src='/img/gli-banner.png' />
</p>


# Graph Learning Indexer (GLI)

[![Pycodestyle](https://github.com/Graph-Learning-Benchmarks/gli/actions/workflows/pycodestyle.yml/badge.svg)](https://github.com/Graph-Learning-Benchmarks/gli/actions/workflows/pycodestyle.yml)
[![Pydocstyle](https://github.com/Graph-Learning-Benchmarks/gli/actions/workflows/pydocstyle.yml/badge.svg)](https://github.com/Graph-Learning-Benchmarks/gli/actions/workflows/pydocstyle.yml)
[![Pylint](https://github.com/Graph-Learning-Benchmarks/gli/actions/workflows/pylint.yml/badge.svg)](https://github.com/Graph-Learning-Benchmarks/gli/actions/workflows/pylint.yml)
[![Pytest](https://github.com/Graph-Learning-Benchmarks/gli/actions/workflows/pytest.yml/badge.svg)](https://github.com/Graph-Learning-Benchmarks/gli/actions/workflows/pytest.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2212.04537-<COLOR>.svg)](https://arxiv.org/abs/2212.04537)

Graph Learning Indexer (GLI) is a benchmark curation platform for graph learning.

## Design Objectives
In comparison to previous graph learning libraries, GLI highlights two design objectives.

* GLI is designed to better serve **dataset contributors** by minimizing the effort of contributing and maintaining a dataset.
* GLI is designed to create a knowledge base (as opposed to a simple
  collection) of benchmarks with **rich meta-information** about the datasets.
  See the [GLI meta-info page](https://gli-vis.streamlit.app/) for details.

## Highlighted Features

### File-Based Data API

GLI defines a file-based standard dataset API that is both efficient in storage and flexible for various graph structures. In comparison to the common code-based dataset API, the file-based design can significantly reduce the maintenance effort required for the dataset contributors.

### Explicit Separation of Data and Task

GLI makes an explicit separation between the data storage and the task configuration. For graph learning, there could often be multiple tasks (e.g., node classification and link prediction) defined on the same dataset, or there could be multiple settings for the same task (e.g., random split or fixed split).

The explicit separation of data and task provides a number of benefits:

- The API becomes more extensible to new tasks.
- The automated tests can be separated by tasks and become more modularized.
- It allows implementation of general data loading schemes for each task.


### Automated Tests

GLI implements a wide range of automated tests for new dataset submissions, which provides prompt and rich feedback to the dataset contributors and makes the contribution process smoother.


### Rich Meta Information

GLI also provides tools to calculate graph properties (such as clustering coefficients or homophily ratio) and benchmark popular models for newly contributed datasets, which can augment new datasets with rich meta-information.


<!-- TODO: Add more highlighted features. -->

## Get Started

This is a quick start for users who want to use the existing datasets hosted in GLI. For users who want to contribute a new dataset, please refer to our [Contribution Guide](./CONTRIBUTING.md).

### Installation

Currently, we support installation from the source.

```bash
git clone https://github.com/Graph-Learning-Benchmarks/gli.git
cd gli
pip install -e .
```
> *Note: [wget](https://www.gnu.org/software/wget/) is required to download datasets.*

To test the installation, run the following command:

```bash
python example.py --graph cora --task NodeClassification
```

The output should be something like the following:

```
> Graph(s) loading takes 0.0196 seconds and uses 0.9788 MB.
> Task loading takes 0.0016 seconds and uses 0.1218 MB.
> Combining(s) graph and task takes 0.0037 seconds and uses 0.0116 MB.
Dataset("CORA dataset. NodeClassification", num_graphs=1, save_path=~/.dgl/CORA dataset. NodeClassification)**
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

All kinds of improvement are welcomed! Please refer to our [Contribution Guide](./CONTRIBUTING.md) for details.


## Citation

**Note**: If you are using a dataset hosted in `datasets/`, please cite the corresponding data source listed in the README.md of *that dataset*.

If you find GLI helpful for your research, please consider citing our paper below.

[Graph Learning Indexer: A Contributor-Friendly and Metadata-Rich Platform for Graph Learning Benchmarks](https://openreview.net/pdf?id=ZBsxA6_gp3).

Jiaqi Ma*, Xingjian Zhang*, Hezheng Fan, Jin Huang, Tianyue Li, Ting Wei Li, Yiwen Tu, Chenshu Zhu, and Qiaozhu Mei. LOG 2022. (*Equal Contributions.)

BibTex:
```
@inproceedings{ma2022graph,
      title={Graph Learning Indexer: A Contributor-Friendly and Metadata-Rich Platform for Graph Learning Benchmarks},
      author={Jiaqi Ma and Xingjian Zhang and Hezheng Fan and Jin Huang and Tianyue Li and Ting Wei Li and Yiwen Tu and Chenshu Zhu and Qiaozhu Mei},
      booktitle={The First Learning on Graphs Conference},
      year={2022},
      url={https://openreview.net/forum?id=ZBsxA6_gp3}
}
```
