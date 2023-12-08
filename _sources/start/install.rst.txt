Installation
===============================

Currently, we support installation from the source.

.. code:: bash

   git clone https://github.com/Graph-Learning-Benchmarks/gli.git
   cd gli
   pip install -e .           # basic requirements
   pip install -e ".[test]"   # test-related requirements
   pip install -e ".[doc]"    # doc-related requirements
   pip install -e ".[full]"   # all requirements

To test the installation, run the following command:

.. code:: bash

   python example.py --graph cora --task NodeClassification

The output should be like this:

::

   > Graph(s) loading takes 0.0196 seconds and uses 0.9788 MB.
   > Task loading takes 0.0016 seconds and uses 0.1218 MB.
   > Combining(s) graph and task takes 0.0037 seconds and uses 0.0116 MB.
   Dataset("CORA dataset. NodeClassification", num_graphs=1, save_path=/Users/jimmy/.dgl/CORA dataset. NodeClassification)**