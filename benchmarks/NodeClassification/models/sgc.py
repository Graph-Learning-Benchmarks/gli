"""
SGConv model in GLI.

References:
https://docs.dgl.ai/generated/dgl.nn.pytorch.conv.SGConv.html
https://github.com/dmlc/dgl/blob/master/examples/pytorch/sgc/sgc.py
"""

from torch import nn
from dgl.nn.pytorch import SGConv


class SGC(nn.Module):
    """SGC network."""

    def __init__(self,
                 g,
                 in_feats,
                 n_classes,
                 k) -> None:
        """Initiate model."""
        super().__init__()
        self.g = g
        self.layer = SGConv(in_feats, n_classes, k)

    def forward(self, features):
        """Forward."""
        h = features
        h = self.layer(self.g, h)
        return h
