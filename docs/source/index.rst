.. GLI documentation master file, created by
   sphinx-quickstart on Sun Oct 30 13:29:10 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GLI's Tutorial and Documentation!
============================================

GLI is an easy-to-use graph learning platform with unique features that can better serve the dataset contributors, in comparison to existing graph learning libraries. It aims to ease and incentivize the creation and curation of datasets.

Highlighted Features
--------------------

Standard Data Format
~~~~~~~~~~~~~~~~~~~~

GLI defines a standard data format that has efficient storage and access to graphs. It unifies the storage for graphs of different scales and heterogeneity and is thus flexible to accommodate various graph-structured data.

Explicit Separation of Data Storage and Task Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GLI makes an explicit separation between the data storage and the task configuration for graph learning. i.e., Multiple tasks can be performed on the same dataset, or the same task can be performed on different datasets. The separation between graphs and tasks further allows users to use general datasets bound to every type of task that can be applied to every graph dataset.

.. toctree::
   :maxdepth: 1
   :caption: Get Started
   :glob:

   start/install
   start/tutorial
   start/contribute

.. toctree::
   :maxdepth: 2
   :caption: Modules

   modules

.. toctree:: 
   :maxdepth: 2
   :caption: Data

   format

