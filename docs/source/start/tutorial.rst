Tutorial
===============================
Data Loading API
~~~~~~~~~~~~~~~~

Welcome to the tutorial! Let's start with the data loading API that is
used to assemble a dataset from the given graph(s) and task. To load a
dataset from the remote data repository, simply use the
:func:`gli.dataloading.get_gli_dataset` function:

.. code:: python

   >>> import gli
   >>> dataset = gli.get_gli_dataset(dataset="cora", task="NodeClassification", device="cpu")
   >>> dataset
   Dataset("CORA dataset. NodeClassification", num_graphs=1, save_path=/Users/jimmy/.dgl/CORA dataset. NodeClassification)

The above code loads the Cora on :class:`gli.task.NodeClassificationTask` that is
predefined in the GLI repository. :func:`gli.dataloading.get_gli_dataset` essentially
does three things: 

1. Load the requested graph(s). 
2. Load the requested task configuration. 
3. Combine them to return a dataset instance.

Alternatively, one can do the same thing step by step, with the help of
functions provided by GLI. 

1. :func:`gli.dataloading.get_gli_graph`, :func:`gli.graph.read_gli_graph`.
2. :func:`gli.dataloading.get_gli_task`, :func:`gli.task.read_gli_task`.
3. :func:`gli.dataloading.combine_graph_and_task`.

In specific, methods started with ``get`` will download data from the remote repository and methods started with ``read`` will read data files from local directories.
GLI adopts the graph classes of DGL. Therefore, :func:`gli.dataloading.get_gli_graph` will return a ``DGLGraph`` instance, or a list of ``DGLGraph`` if the dataset
contains multiple graphs. Besides, GLI provides class implementations for various tasks (e.g., :class:`gli.task.NodeClassificationTask`, :class:`gli.task.LinkPredictionTask`).
Furthermore, :func:`gli.dataloading.get_gli_task` will return a :class:`gli.task.GLITask` object. One can then call :func:`gli.dataloading.combine_graph_and_task` to assemble
a corresponding dataset (e.g., :class:`gli.dataset.NodeClassificationDataset`, :class:`gli.dataset.LinkPredictionDataset`).

.. code:: python

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

The returned dataset is inherited from ``DGLDataset``. Therefore, it can
be incorporated into DGL's infrastructure seamlessly:

.. code:: python

   >>> type(dataset)
   <class 'gli.dataset.NodeClassificationDataset'>
   >>> isinstance(dataset, dgl.data.DGLDataset)
   True

Example
~~~~~~~

Next, let's see a full example of dataloading and training on GLI datasets.