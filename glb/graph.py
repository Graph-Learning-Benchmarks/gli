import json
import os
from typing import Iterable, List, Union

import numpy as np
import scipy as sp
import torch
import pandas as pd
from scipy import sparse


class Graph(object):
    """Base graph class for dataset loading."""

    def __init__(self, edges, node_list, edge_list, node_attrs, edge_attrs, graph_attrs, description, citation):
        """Graph class does not distinguish between features and labels.
        They are treated equally in the class."""
        self.edges = edges
        self.nodelist = node_list
        self.edgelist = edge_list

        self.node_attrs = node_attrs
        self.edge_attrs = edge_attrs
        self.graph_attrs = graph_attrs

        self.descirption = description
        self.citation = citation

        print(self.descirption)
        print("\t#nodes: ", self.num_nodes)
        print("\t#edges: ", self.num_edges)

    @staticmethod
    def load_graph(meta_path: os.PathLike):
        """Initialize and return a Graph instance given metadata.json."""
        pwd = os.path.dirname(meta_path)
        with open(meta_path, 'r') as fp:
            metadata = json.load(fp)

        description = metadata.get("description", None)
        citation = metadata.get("citation", None)

        assert "data" in metadata, "attribute `data` not in metadata.json."

        for neg in ["Node", "Edge", "Graph"]:
            assert neg in metadata["data"], "attribute `{}` not in metadata.json".format(
                neg)

        assert "_Edge" in metadata["data"]["Edge"]
        assert "_NodeList" in metadata["data"]["Graph"]

        data_buffer = {}
        data = {"Node": {}, "Edge": {}, "Graph": {}}
        for neg in ["Node", "Edge", "Graph"]:
            for attr, props in metadata["data"][neg].items():
                if "file" in props:
                    filename = props["file"]
                    if filename not in data_buffer:
                        raw = load_data(os.path.join(pwd, filename))
                        data_buffer[filename] = raw
                    else:
                        raw = data_buffer[filename]
                    if "key" in props:
                        key = props["key"]
                        _raw = raw[key]
                    else:
                        _raw = raw
                    feat = Feature(name=attr,
                                   desc=props.get("description"),
                                   type=props.get("type"),
                                   format=props.get("format"),
                                   data=_raw)
                    data[neg][attr] = feat

        # collate features into graph
        edges = data["Edge"].pop("_Edge")
        node_list = data["Graph"].pop("_NodeList")
        edge_list = data["Graph"].pop("_EdgeList", None)

        node_attrs = data["Node"]
        edge_attrs = data["Edge"]
        graph_attrs = data["Graph"]

        return Graph(edges, node_list, edge_list, node_attrs, edge_attrs, graph_attrs, description, citation)

    @property
    def num_nodes(self):
        return self.nodelist.data.shape[-1]
    
    @property
    def num_edges(self):
        return self.edges.data.shape[0]

def load_data(path: os.PathLike):
    _, ext = os.path.splitext(path)
    if ext == ".npz":
        data = np.load(path, allow_pickle=True)
    else:
        raise NotImplementedError(
            "{} file is currently not supported.".format(ext))
    return data


class Feature(object):
    """Base class Feature for all feature attributes in dataset loading."""

    def __init__(self, name, desc, type, format, data) -> None:
        self.name = name
        self.description = desc
        self.data = data
        self.format = format
        self.type = type

    def __repr__(self) -> str:
        if self.description:
            desc = self.description
        else:
            desc = ""
        s = "<Feature> {}: {}\n\t{}".format(self.name, desc, repr(self.data))
        return s