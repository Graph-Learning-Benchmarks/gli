"""
TAGCN model in GLI.

References:
https://docs.dgl.ai/generated/dgl.nn.pytorch.conv.TAGConv.html
"""

from dgl.nn.pytorch.conv import TAGConv
from torch import nn


class TAGCN(nn.Module):
    """TAGCN network."""

    def __init__(
        self,
        g,
        in_feats,
        n_hidden,
        n_classes,
        n_layers,
        k,
        activation,
        dropout
    ):
        """Initiate model."""
        super().__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(TAGConv(in_feats,
                                   n_hidden,
                                   k=k,
                                   activation=activation))
        # hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(
                TAGConv(n_hidden, n_hidden, activation=activation)
            )
        # output layer
        self.layers.append(TAGConv(n_hidden, n_classes))  # activation=None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        """Forward."""
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h
