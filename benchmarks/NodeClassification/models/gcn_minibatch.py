"""
GCN model in GLB.

References:
https://github.com/dmlc/dgl/tree/master/examples/pytorch/gcn
https://docs.dgl.ai/guide/minibatch-node.html?highlight=sampling
"""

from torch import nn
from dgl.nn.pytorch import GraphConv


class GCN_minibatch(nn.Module):
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
        self.blocks = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden,
                                     activation=activation,
                                     norm='none'))
        # hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden,
                                         activation=activation,
                                         norm='none'))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes,
                                     norm='none'))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        """Forward."""
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.blocks[i], h)
        return h

