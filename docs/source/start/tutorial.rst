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

First, import all required modules.

.. code:: python

   import gli
   import torch
   from torch import nn
   import torch.nn.functional as F
   from dgl.nn.pytorch import GraphConv
   from gli.utils import to_dense

Then, load the Cora dataset on node classification task.

.. code:: python

   data = gli.dataloading.get_gli_dataset("cora", "NodeClassification")
   g = data[0]
   g = to_dense(g)

Since there are sparse features in Cora dataset, we need to convert it to dense for later computation.

We then define the evaluation function as below.

.. code:: python

   def accuracy(logits, labels):
      """Calculate accuracy."""
      _, indices = torch.max(logits, dim=1)
      correct = torch.sum(indices == labels)
      return correct.item() * 1.0 / len(labels)


   def evaluate(model, features, labels, mask, eval_func):
      """Evaluate model."""
      model.eval()
      with torch.no_grad():
         logits = model(features)
         logits = logits[mask]
         labels = labels[mask]
         return eval_func(logits, labels)

Next, we define a GCN model and start training.

.. code:: python

   class GCN(nn.Module):
      """GCN network."""

      def __init__(self,
                  g,
                  in_feats,
                  n_hidden,
                  n_classes,
                  n_layers,
                  activation,
                  dropout):
         """Initiate model."""
         super().__init__()
         self.g = g
         self.layers = nn.ModuleList()
         # input layer
         self.layers.append(GraphConv(in_feats, n_hidden,
                                       activation=activation))
         # hidden layers
         for _ in range(n_layers - 2):
               self.layers.append(GraphConv(n_hidden, n_hidden,
                                          activation=activation))
         # output layer
         self.layers.append(GraphConv(n_hidden, n_classes))
         self.dropout = nn.Dropout(p=dropout)

      def forward(self, features):
         """Forward."""
         h = features
         for i, layer in enumerate(self.layers):
               if i != 0:
                  h = self.dropout(h)
               h = layer(self.g, h)
         return h

   model = GCN(g=g,
               in_feats=in_feats,
               n_hidden=8,
               n_classes=n_classes,
               n_layers=2,
               activation=F.relu,
               dropout=.6)

   optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.001)
   eval_func = accuracy
   loss_fcn = nn.CrossEntropyLoss()

   for epoch in range(200):
         model.train()

         # forward
         logits = model(features)
         loss = loss_fcn(logits[train_mask], labels[train_mask])

         optimizer.zero_grad()
         loss.backward()
         optimizer.step()

         train_acc = eval_func(logits[train_mask], labels[train_mask])
         val_acc = evaluate(model, features, labels, val_mask, eval_func)
         print(f"Epoch {epoch:05d} | Loss {loss.item():.4f} |"
               f"TrainAcc {train_acc:.4f} | ValAcc {val_acc:.4f}")

   test_acc = evaluate(model, features, labels, test_mask, eval_func)
   print(f"Test Accuracy: {test_acc:.4f}")

Output:

.. code:: text

   Epoch 00000 | Loss 1.9454 |TrainAcc 0.1429 | ValAcc 0.3180
   Epoch 00001 | Loss 1.9375 |TrainAcc 0.2500 | ValAcc 0.3580
   Epoch 00002 | Loss 1.9318 |TrainAcc 0.3286 | ValAcc 0.3940
   Epoch 00003 | Loss 1.9242 |TrainAcc 0.3357 | ValAcc 0.4100
   Epoch 00004 | Loss 1.9138 |TrainAcc 0.4214 | ValAcc 0.4420
   Epoch 00005 | Loss 1.9039 |TrainAcc 0.5143 | ValAcc 0.4720
   Epoch 00006 | Loss 1.9002 |TrainAcc 0.4143 | ValAcc 0.4740
   Epoch 00007 | Loss 1.8891 |TrainAcc 0.4643 | ValAcc 0.4660
   Epoch 00008 | Loss 1.8787 |TrainAcc 0.5071 | ValAcc 0.4760
   Epoch 00009 | Loss 1.8733 |TrainAcc 0.4286 | ValAcc 0.5020
   Epoch 00010 | Loss 1.8581 |TrainAcc 0.5857 | ValAcc 0.5280
   ...