"""
MIXHOP model in GLI.

References:
https://github.com/dmlc/dgl/tree/master/examples/pytorch/mixhop
"""

import torch
from torch import nn
import torch.nn.functional as F
import dgl.function as fn


class MixHopConv(nn.Module):
    r"""MixHopConv layer."""

    def __init__(self,
                 in_dim,
                 out_dim,
                 p,
                 dropout=0,
                 activation=None,
                 batchnorm=False):
        """Initiate layer."""
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.p = p
        self.activation = activation
        self.batchnorm = batchnorm

        # define dropout layer
        self.dropout = nn.Dropout(dropout)

        # define batch norm layer
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim * len(p))

        # define weight dict for each power j
        self.weights = nn.ModuleDict({
            str(j): nn.Linear(in_dim, out_dim, bias=False) for j in p
        })

    def forward(self, graph, feats):
        """Forward."""
        with graph.local_scope():
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).to(feats.device).unsqueeze(1)
            max_j = max(self.p) + 1
            outputs = []
            for j in range(max_j):

                if j in self.p:
                    output = self.weights[str(j)](feats)
                    outputs.append(output)

                feats = feats * norm
                graph.ndata['h'] = feats
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                feats = graph.ndata.pop('h')
                feats = feats * norm

            final = torch.cat(outputs, dim=1)

            if self.batchnorm:
                final = self.bn(final)

            if self.activation is not None:
                final = self.activation(final)

            final = self.dropout(final)

            return final


class MixHop(nn.Module):
    """MixHop model."""

    def __init__(self,
                 g,
                 in_dim,
                 hid_dim,
                 out_dim,
                 p,
                 num_layers=2,
                 input_dropout=0.0,
                 layer_dropout=0.0,
                 batchnorm=False):
        """Initiate model."""
        super().__init__()
        self.g = g
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.p = p
        self.input_dropout = input_dropout
        self.layer_dropout = layer_dropout
        self.activation = F.tanh
        self.batchnorm = batchnorm

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(self.input_dropout)

        # Input layer
        self.layers.append(MixHopConv(self.in_dim,
                                      self.hid_dim,
                                      p=self.p,
                                      dropout=self.input_dropout,
                                      activation=self.activation,
                                      batchnorm=self.batchnorm))

        # Hidden layers with n - 1 MixHopConv layers
        for _ in range(self.num_layers - 2):
            self.layers.append(MixHopConv(self.hid_dim * len(p),
                                          self.hid_dim,
                                          p=self.p,
                                          dropout=self.layer_dropout,
                                          activation=self.activation,
                                          batchnorm=self.batchnorm))

        self.fc_layers = nn.Linear(self.hid_dim * len(p),
                                   self.out_dim, bias=False)

    def forward(self, feats):
        """Forward."""
        feats = self.dropout(feats)
        for layer in self.layers:
            feats = layer(self.g, feats)

        feats = self.fc_layers(feats)
        return feats
