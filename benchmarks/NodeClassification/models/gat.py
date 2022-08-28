"""
GAT model in GLI.

References:
https://github.com/dmlc/dgl/blob/master/examples/pytorch/gat/gat.py
"""

from torch import nn
from dgl.nn import GATConv


class GAT(nn.Module):
    """GAT network."""

    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        """Initiate model."""
        super().__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for layer in range(1, num_layers-1):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(num_hidden * heads[layer-1],
                                           num_hidden, heads[layer],
                                           feat_drop, attn_drop,
                                           negative_slope, residual,
                                           self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs):
        """Forward."""
        h = inputs
        for layer in range(self.num_layers):
            h = self.gat_layers[layer](self.g, h)
            h = h.flatten(1) if layer != self.num_layers - 1 else h.mean(1)
        return h
