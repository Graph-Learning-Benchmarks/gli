"""GCN model in GLB."""
from torch import nn
from dgl.nn.pytorch import GraphConv


class GCN(nn.Module):
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
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden,
                                     activation=activation))
        # hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden,
                                         activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        """Forward."""
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h
