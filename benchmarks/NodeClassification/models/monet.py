"""
GAT model in GLB.

References:
https://github.com/dmlc/dgl/blob/master/examples/pytorch/monet/citation.py
"""

from torch import nn
from dgl.nn.pytorch.conv import GMMConv


class MoNet(nn.Module):
    """Monet model."""

    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 out_feats,
                 n_layers,
                 dim,
                 n_kernels,
                 dropout):
        """Initiate model."""
        super().__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.pseudo_proj = nn.ModuleList()

        # Input layer
        self.layers.append(
            GMMConv(in_feats, n_hidden, dim, n_kernels))
        self.pseudo_proj.append(
            nn.Sequential(nn.Linear(2, dim), nn.Tanh()))

        # Hidden layer
        for _ in range(n_layers - 1):
            self.layers.append(GMMConv(n_hidden, n_hidden, dim, n_kernels))
            self.pseudo_proj.append(
                nn.Sequential(nn.Linear(2, dim), nn.Tanh()))

        # Output layer
        self.layers.append(GMMConv(n_hidden, out_feats, dim, n_kernels))
        self.pseudo_proj.append(
            nn.Sequential(nn.Linear(2, dim), nn.Tanh()))
        self.dropout = nn.Dropout(dropout)

    def forward(self, feat, pseudo):
        """Forward."""
        h = feat
        for i in range(len(self.layers)):
            if i != 0:
                h = self.dropout(h)
            h = self.layers[i](
                self.g, h, self.pseudo_proj[i](pseudo))
        return h
