"""
GraphSAGE model in GLI.

References:
https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/train_full.py
https://docs.dgl.ai/guide/minibatch-node.html?highlight=sampling
"""

from torch import nn
from dgl.nn.pytorch.conv import SAGEConv


class GraphSAGEminibatch(nn.Module):
    """GraphSAGE model."""

    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        """Initiate model."""
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden,
                           aggregator_type))
        # hidden layers
        for _ in range(n_layers - 2):
            self.layers.append(SAGEConv(n_hidden, n_hidden,
                               aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes,
                           aggregator_type))

    def forward(self, blocks, inputs):
        """Forward."""
        h = self.dropout(inputs)
        for length, layer in enumerate(self.layers):
            h = layer(blocks[length], h)
            if length != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h
