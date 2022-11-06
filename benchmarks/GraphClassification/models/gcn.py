"""
GCN model in GLI.

References:
https://github.com/dmlc/dgl/tree/master/examples/pytorch/gcn
https://docs.dgl.ai/tutorials/blitz/5_graph_classification.html#
sphx-glr-tutorials-blitz-5-graph-classification-py
"""

import dgl
from torch import nn
from dgl.nn.pytorch import GraphConv
from models.mlp_readout_layer import MLPReadout


class GCNgraph(nn.Module):
    """GCN network."""

    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        """Initiate model."""
        super().__init__()
        self.layers = nn.ModuleList()
        # embedded layer
        self.embedding_h = nn.Linear(in_feats, n_hidden)

        # hidden layers
        for _ in range(n_layers - 2):
            self.layers.append(GraphConv(n_hidden, n_hidden,
                                         activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_hidden))
        self.dropout = nn.Dropout(p=dropout)

        # readout layer
        self.mlp_layer = MLPReadout(n_hidden, n_classes)

    def forward(self, g, features):
        """Forward."""
        h = features
        h = self.embedding_h(h)
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        g.ndata["h"] = h
        hg = dgl.mean_nodes(g, "h")
        return self.mlp_layer(hg)
