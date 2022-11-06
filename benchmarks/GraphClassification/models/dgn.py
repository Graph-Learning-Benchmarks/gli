"""
DGN model in GLI.

References:
https://docs.dgl.ai/generated/dgl.nn.pytorch.conv.DGNConv.html?
highlight=dgn#dgl.nn.pytorch.conv.DGNConv
https://docs.dgl.ai/tutorials/blitz/5_graph_classification.html#
sphx-glr-tutorials-blitz-5-graph-classification-py
"""

import dgl
from dgl import LaplacianPE
from dgl.nn.pytorch import DGNConv
from torch import nn


class DGN(nn.Module):
    """GDN network."""

    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 aggregators,
                 scalers,
                 delta,
                 dropout):
        """Initiate model."""
        super().__init__()
        self.layers = nn.ModuleList()
        self.aggregators = aggregators
        # input layer
        self.layers.append(DGNConv(in_feats, n_hidden,
                                   aggregators,
                                   scalers,
                                   delta,
                                   dropout=dropout))
        # hidden layers
        for _ in range(n_layers - 2):
            self.layers.append(DGNConv(n_hidden, n_hidden,
                                       aggregators,
                                       scalers,
                                       delta,
                                       dropout=dropout))
        # output layer
        self.layers.append(DGNConv(n_hidden, n_classes,
                                   aggregators,
                                   scalers,
                                   delta,
                                   dropout=dropout))

    def forward(self, g, features):
        """Forward."""
        h = features
        if [agg for agg in self.aggregators if "dir" in agg]:
            # if aggregator contains directional ones
            transform = LaplacianPE(k=3, feat_name="eig")
            g = transform(g)
            eig = g.ndata["eig"]
            for layer in self.layers:
                h = layer(g, h, eig_vec=eig)
        else:
            for layer in self.layers:
                h = layer(g, h)
        g.ndata["h"] = h
        return dgl.mean_nodes(g, "h")
