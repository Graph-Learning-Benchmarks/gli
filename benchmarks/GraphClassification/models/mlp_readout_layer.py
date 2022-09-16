
"""
MLP Layer used after graph vector representation.

References:
https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/layers/mlp_readout_layer.py
"""

from torch import nn
import torch.nn.functional as F


class MLPReadout(nn.Module):
    """MLPReadout layer in GLI."""

    def __init__(self, input_dim, output_dim, L=2):
        """Initiate layer, L=nb_hidden_layers."""
        super().__init__()
        list_fc_layers = [nn.Linear(input_dim // 2 ** layer, input_dim // 2 **
                          (layer + 1), bias=True) for layer in range(L)]
        list_fc_layers.append(nn.Linear(input_dim // 2 ** L, output_dim,
                              bias=True))
        self.fc_layers = nn.ModuleList(list_fc_layers)
        self.n_layers = L

    def forward(self, x):
        """Forward."""
        y = x
        for layer in range(self.n_layers):
            y = self.fc_layers[layer](y)
            y = F.relu(y)
        y = self.fc_layers[self.n_layers](y)
        return y
