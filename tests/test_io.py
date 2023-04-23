import tempfile
import os

import numpy as np
import pytest
from numpy.random import randint, randn
from scipy.sparse import random as sparse_random
from scipy.sparse import csr_matrix

import gli
import gli.io
from gli.io import Attribute


def test_save_homograph_wo_graph_lists():
    """Test saving and loading a homograph.

    Create a temporary dir and save a homograph to it.
    Then load it back and compare the data.
    """
    # Create a temporary dir.
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a graph with 6 nodes and 5 edges
        edge = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
        # Create attributes of the nodes.
        dense_node_feats = Attribute(
            name="DenseNodeFeature",
            data=randn(6, 5),  # 6 nodes, 5 features
            description="Dense node features.")
        sparse_node_feats = Attribute(
            name="SparseNodeFeature",
            data=sparse_random(6, 500),  # 6 nodes, 500 features
            description="Sparse node features.")
        node_labels = Attribute(
            name="NodeLabel",
            data=randint(0, 4, 6),  # 6 nodes, 4 classes
            description="Node labels.")
        edge_feats = Attribute(
            name="EdgeAttribute",
            data=randn(5, 3),  # 5 edges, 3 features
            description="Dense edge features.")
        sparse_edge_feats = Attribute(
            name="SparseEdgeAttribute",
            data=sparse_random(5, 500),  # 5 edges, 500 features
            description="Sparse edge features.")
        edge_labels = Attribute(
            name="EdgeLabel",
            data=randint(0, 4, 5),  # 5 edges, 4 classes
            description="Edge labels.")

        # Save the graph dataset.
        gli.io.save_homograph(
            name="example_dataset",
            edge=edge,
            node_attrs=[dense_node_feats, sparse_node_feats, node_labels],
            edge_attrs=[edge_feats, sparse_edge_feats, edge_labels],
            description="An exampmle dataset.",
            citation="some bibtex citation",
            save_dir=tmpdir)

        # Load the graph dataset.
        metadata_path = os.path.join(tmpdir, "metadata.json")
        g = gli.graph.read_gli_graph(metadata_path)


def test_save_homograph_with_graph_node_lists():
    """Test saving and loading a homograph.

    The data to be saved contains a node_graph_list.
    """
    # Create a temporary dir.
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a graph with 6 nodes and 5 edges (2 connected components)
        edge = np.array([[0, 1], [1, 2], [0, 2], [3, 4], [4, 5]])
        # Create attributes of the nodes.
        dense_node_feats = Attribute(
            name="DenseNodeFeature",
            data=randn(6, 5),  # 6 nodes, 5 features
            description="Dense node features.")
        sparse_node_feats = Attribute(
            name="SparseNodeFeature",
            data=sparse_random(6, 500),  # 6 nodes, 500 features
            description="Sparse node features.")
        node_labels = Attribute(
            name="NodeLabel",
            data=randint(0, 4, 6),  # 6 nodes, 4 classes
            description="Node labels.")
        edge_feats = Attribute(
            name="EdgeAttribute",
            data=randn(5, 3),  # 5 edges, 3 features
            description="Dense edge features.")
        sparse_edge_feats = Attribute(
            name="SparseEdgeAttribute",
            data=sparse_random(5, 500),  # 5 edges, 500 features
            description="Sparse edge features.")
        edge_labels = Attribute(
            name="EdgeLabel",
            data=randint(0, 4, 5),  # 5 edges, 4 classes
            description="Edge labels.")
        graph_node_list = np.array([
            [1, 1, 1, 0, 0, 0],  # 3 nodes in the first graph
            [0, 0, 0, 1, 1, 1]  # 3 nodes in the second graph
        ]).astype(np.int32)
        # transform graph_node_list to a scipy sparse matrix
        graph_node_list = csr_matrix(graph_node_list)

        # Save the graph dataset.
        gli.io.save_homograph(
            name="example_dataset",
            edge=edge,
            graph_node_list=graph_node_list,
            node_attrs=[dense_node_feats, sparse_node_feats, node_labels],
            edge_attrs=[edge_feats, sparse_edge_feats, edge_labels],
            description="An exampmle dataset.",
            citation="some bibtex citation",
            save_dir=tmpdir)

        # Load the graph dataset.
        metadata_path = os.path.join(tmpdir, "metadata.json")
        g = gli.graph.read_gli_graph(metadata_path)
        assert len(g) == 2, "The number of graphs should be 2."