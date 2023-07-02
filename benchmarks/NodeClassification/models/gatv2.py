"""
GATv2 model in GLI.

References:
https://docs.dgl.ai/generated/dgl.nn.pytorch.conv.GATv2Conv.html
https://github.com/dmlc/dgl/blob/master/examples/pytorch/gatv2/gatv2.py
"""

from torch import nn
from dgl.nn.pytorch import GATv2Conv


class GATv2(nn.Module):
    """GATv2 network."""

    def __init__(
        self,
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
        residual,
    ):
        """Initiate model."""
        super().__init__()
        self.g = g
        self.num_layers = num_layers
        self.gatv2_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gatv2_layers.append(
            GATv2Conv(
                in_dim,
                num_hidden,
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                self.activation,
                bias=False,
                share_weights=True,
            )
        )
        # hidden layers
        for layer in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gatv2_layers.append(
                GATv2Conv(
                    num_hidden * heads[layer - 1],
                    num_hidden,
                    heads[layer],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                    bias=False,
                    share_weights=True,
                )
            )
        # output projection
        self.gatv2_layers.append(
            GATv2Conv(
                num_hidden * heads[-2],
                num_classes,
                heads[-1],
                feat_drop,
                attn_drop,
                negative_slope,
                residual,
                None,
                bias=False,
                share_weights=True,
            )
        )

    def forward(self, inputs):
        """Forward."""
        h = inputs
        for layer in range(self.num_layers):
            h = self.gatv2_layers[layer](self.g, h).flatten(1)
        # output projection
        logits = self.gatv2_layers[-1](self.g, h).mean(1)
        return logits
