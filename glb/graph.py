"""Base class of graph.

Any dataset needs to contain a graph instance. By default, a graph
instance should be initialized given by a metadata.json.
"""
import json
import os

import torch

from .utils import load_data, is_raw_sparse


class Graph:
    """Base graph class for dataset loading."""

    def __init__(self, **kwargs):
        """Initialize graph."""
        valid_keys = [
            "edges", "node_list", "edge_list", "node_attrs", "edge_attrs",
            "graph_attrs", "description", "citation"
        ]
        for key in valid_keys:
            setattr(self, key, kwargs.get(key))

        print(getattr(self, "description"))
        print("  #Nodes: ", self.num_nodes)
        print("  #Edges: ", self.num_edges)

    @staticmethod
    def load_graph(metadata_path: os.PathLike, device="cpu"):
        """Initialize and return a Graph instance given metadata.json."""
        pwd = os.path.dirname(metadata_path)
        with open(metadata_path, "r", encoding="utf-8") as fptr:
            metadata = json.load(fptr)

        assert "data" in metadata, "attribute `data` not in metadata.json."

        for neg in ["Node", "Edge", "Graph"]:
            assert neg in metadata[
                "data"], f"attribute `{neg}` not in metadata.json"

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
                        array = raw[key]
                    else:
                        array = raw
                    # TODO - consider sparse case
                    if is_raw_sparse(array):
                        array = array.all().toarray()
                    array = torch.from_numpy(array).to(device=device)
                    data[neg][attr] = Feature(name=attr,
                                              desc=props.get("description"),
                                              dtype=props.get("type"),
                                              dformat=props.get("format"),
                                              data=array)

        # collate features into graph
        init_dict = {
            "edges": data["Edge"].pop("_Edge").to_type(Edges),
            "node_list": data["Graph"].pop("_NodeList").to_type(NodeList),
            "edge_list": data["Graph"].pop("_EdgeList", None),
            "node_attrs": data["Node"],
            "edge_attrs": data["Edge"],
            "graph_attrs": data["Graph"],
            "citation": metadata.get("citation", None),
            "description": metadata.get("description", None)
        }
        if init_dict["edge_list"]:
            init_dict["edge_list"] = init_dict["edge_list"].to_type(EdgeList)

        return Graph(**init_dict)

    @property
    def num_nodes(self):
        """Return number of nodes in the graph."""
        return getattr(self, "node_list").data.shape[-1]

    @property
    def num_edges(self):
        """Return number of edges in the graph."""
        return getattr(self, "edges").data.shape[0]


class Feature:
    """Base class Feature for all feature attributes in dataset loading."""

    def __init__(self, **kwargs) -> None:
        """Initialize Feature."""
        valid_keys = ["name", "description", "dtype", "dformat", "data"]
        for key in valid_keys:
            setattr(self, key, kwargs.get(key))

    def __repr__(self) -> str:
        """Represent Feature."""
        desc = getattr(self, "description", "")
        name = getattr(self, "name")
        data = repr(getattr(self, "data"))
        repr_str = f"<{type(self).__name__}> {name}: {desc}\n\t{data}"
        return repr_str

    def to_type(self, cls):
        """Return a proper Feature subclass: Edges, NodeList, or EdgeList."""
        return cls(data=getattr(self, "data"))


class Edges(Feature):
    """Reserved attributes for edges matrix."""

    def __init__(self, data) -> None:
        """Initialize Edges."""
        name = "_Edge"
        desc = "Edge matrix of graph, (2 x #edges)"
        dtype = "int"
        dformat = "Tensor"
        super().__init__(name=name,
                         description=desc,
                         type=dtype,
                         format=dformat,
                         data=data)


class NodeList(Feature):
    """Reserved attributes for NodeList."""

    def __init__(self, data) -> None:
        """Initialize NodeList."""
        name = "_NodeList"
        desc = "Node list of graph(s), (#graphs x #nodes)"
        dtype = "int"
        dformat = "SparseTensor"
        super().__init__(name=name,
                         description=desc,
                         type=dtype,
                         format=dformat,
                         data=data)


class EdgeList(Feature):
    """Reserved attributes for NodeList."""

    def __init__(self, data) -> None:
        """Initialize EdgeList."""
        name = "_EdgeList"
        desc = "Edge list of graph(s), (#graphs x #edges)"
        dtype = "int"
        dformat = "SparseTensor"
        super().__init__(name=name,
                         description=desc,
                         type=dtype,
                         format=dformat,
                         data=data)
