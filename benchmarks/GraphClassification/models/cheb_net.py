"""
ChebNet model in GLI.

References:
https://github.com/dmlc/dgl/blob/195f99362d883f8b6d131b70a7868a
537e55b786/examples/pytorch/model_zoo/citation_network/models.py
"""

import dgl
from torch import nn
from dgl.nn.pytorch import ChebConv
from models.mlp_readout_layer import MLPReadout


class ChebNet(nn.Module):
    """ChebNet network."""

    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 k):
        """Initiate model."""
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            ChebConv(in_feats, n_hidden, k)
        )
        for _ in range(n_layers - 2):
            self.layers.append(
                ChebConv(n_hidden, n_hidden, k)
            )

        self.layers.append(
            ChebConv(n_hidden, n_hidden, k)
        )
        self.mlp_layer = MLPReadout(n_hidden, n_classes)

    def forward(self, g, features):
        """Forward."""
        h = features
        for layer in self.layers:
            h = layer(g, h)
        g.ndata["h"] = h
        hg = dgl.mean_nodes(g, "h")
        return self.mlp_layer(hg)
