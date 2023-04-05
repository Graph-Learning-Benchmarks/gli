import json
from gli.utils import save_data
import numpy as np
import scipy.sparse as sp

class Attributes(object):
    def __init__(
        self,
        name,
        description=None,
        data=None,
        data_type=None,
        data_format=None
    ):
        self.name = name
        self.description = description
        self.data = data
        self.key = None
        print("data_type: ", data_type)
        print(type(data))
        if type is not None:
            self.type = data_type
        else:
            self.type = type(data)
        if data_format is not None:
            self.format = data_format
        else:
            if isinstance(data, np.ndarray):
                self.format = "Tensor"
            elif sp.issparse(data):
                self.format = "SparseTensor"
    def to_dict(self, loc : dict()):
        if not loc:
            return None
        info = dict()
        if self.description:
            info.update({"description" : self.description})
        # if self.type:
        info.update({"type" : self.type})
        # if self.format:
        info.update({"format" : self.format})
        info.update({"file" : loc})
        info.update({"key" : self.key})
        return {self.name : info}
    def get_key(self, prefix):
        # heterogeneous
        # self.key = "%s/%s" % (prefix, self.name)
        if not self.key:
            self.key = f"{prefix}_{self.name}" #TODO:
        return self.key

def save_graph(name,
    edges,
    node_list,
    edge_list=None,
    node_attrs=[],
    edge_attrs=[],
    graph_attrs=[],
    citation=None,
    is_heterogeneous = False,
):
    # save data
    data = dict()
    for e in edge_attrs:
        data.update({e.get_key("Edge") : e.data})
    for n in node_attrs:
        data.update({n.get_key("Node") : n.data})
    data.update({"edge" : edges})
    data.update({"node_list" : node_list})
    if edge_list is not None:
        data.update({"edge_list" : edge_list})
    key_to_loc = save_data(f"{name}__graph", **data)
    # save metadata.json
    def _attrs_to_dict(attrs):
        d = dict()
        for a in attrs:
            d.update(a.to_dict(key_to_loc.get(a.key)))
        return d
    node = _attrs_to_dict(node_attrs)
    edge = {"_Edge": key_to_loc.get("edge")}
    edge.update(_attrs_to_dict(edge_attrs))
    graph = {"_NodeList": key_to_loc.get("node_list")}
    if edge_list is not None:
        graph.update({"_EdgeList": key_to_loc.get("edge_list")})
    graph.update(_attrs_to_dict(graph_attrs))
    metadata = {"description": "%s dataset" % name,
         "Node": node,
         "Edge": edge,
         "Graph": graph,
         "citation": citation,
         "is_heterogeneous": is_heterogeneous}
    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)