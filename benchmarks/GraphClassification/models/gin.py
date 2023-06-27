"""
GIN model in gli.

References:
https://github.com/dmlc/dgl/tree/master/examples/pytorch/gin
"""

from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling


class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        """Initiate model."""
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        """Forward."""
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)


class GIN(nn.Module):
    """GIN network."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        """Initiate model."""
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        # five-layer GCN with two-layer MLP aggregator
        # and sum-neighbor-pooling scheme
        for layer in range(num_layers - 1):
            # excluding the input layer
            if layer == 0:
                mlp = MLP(input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            # set to True if learning epsilon
            self.ginlayers.append(GINConv(mlp, learn_eps=False))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linear_prediction.append(nn.Linear(input_dim,
                                                        output_dim))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim,
                                                        output_dim))
        self.drop = nn.Dropout(dropout)
        # change to mean readout (AvgPooling) on social network datasets
        self.pool = SumPooling()

    def forward(self, g, h):
        """Forward."""
        # list of hidden representation at each layer,
        # including the input layer
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = 0
        # perform graph sum pooling over all nodes in each layer
        for i, hid in enumerate(hidden_rep):
            pooled_h = self.pool(g, hid)
            score_over_layer += self.drop(self.linear_prediction[i](pooled_h))
        return score_over_layer
