"""
GCNII model in GLI.

References:
https://github.com/chennnM/GCNII/blob/master/model.py

"""

from torch import nn
from dgl.nn.pytorch.conv import GCN2Conv


class GCNII(nn.Module):
    """GCNII model."""

    def __init__(self,
                 g,
                 in_feats,
                 num_hidden,
                 n_classes,
                 num_layers,
                 activation,
                 dropout,
                 lambda_,
                 alpha,):
        """Initiate model."""
        super().__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.fcs.append(nn.Linear(in_feats, num_hidden))
        # hidden layers
        for i in range(num_layers - 2):
            self.layers.append(GCN2Conv(num_hidden, i + 1, alpha, lambda_, activation=activation))
        # output layer
        self.fcs.append(nn.Linear(num_hidden, n_classes))

    def forward(self, inputs):
        """Forward."""
        h = self.activation(self.fcs[0](self.dropout(inputs)))
        h_ = h
        for _, layer in enumerate(self.layers):
            self.activation(layer(self.g, self.dropout(h), h_))
        h = self.activation(self.fcs[-1](self.dropout(h)))
        return h
