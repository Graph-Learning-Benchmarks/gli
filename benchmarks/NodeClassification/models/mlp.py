"""
MLP model in GLI.

References:
https://github.com/dmlc/dgl/blob/195f99362d883f8b6d131b70a7868a537e55b786/examples/pytorch/grand/model.py
"""

from torch import nn


class MLP(nn.Module):
    """MLP network."""

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
        self.activation = activation
        # input layer
        self.layers.append(nn.Linear(in_feats, n_hidden, bias=True))

        # hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden, bias=True))

        # output layer
        self.layers.append(nn.Linear(n_hidden, n_classes, bias=True))

        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        """Forward."""
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h)
            h = self.activation(h)
        return h
