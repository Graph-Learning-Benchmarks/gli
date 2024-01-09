"""
LINKX model in Non-Homophily-Large-Scale.

References:
https://github.com/CUAI/Non-Homophily-Large-Scale
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch_sparse import SparseTensor


class LINKX(nn.Module):
    """
    LINKX method with skip connections.

    a = MLP_1(A), x = MLP_2(X), MLP_3(sigma(W_1[a, x] + a + x)).
    """

    def __init__(self, g, in_channels, hidden_channels, out_channels,
                 num_layers, num_nodes, dropout=.5, inner_activation=False,
                 inner_dropout=False, init_layers_A=1,
                 init_layers_X=1):
        """Initiate model."""
        super().__init__()
        self.g = g
        self.mlpa = MLP(num_nodes, hidden_channels, hidden_channels,
                        init_layers_A, dropout=0)
        self.mlpx = MLP(in_channels, hidden_channels, hidden_channels,
                        init_layers_X, dropout=0)
        self.w = nn.Linear(2*hidden_channels, hidden_channels)
        self.mlp_final = MLP(hidden_channels, hidden_channels, out_channels,
                             num_layers, dropout=dropout)
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.inner_activation = inner_activation
        self.inner_dropout = inner_dropout

    def reset_parameters(self):
        """Reset parameters."""
        self.mlpa.reset_parameters()
        self.mlpx.reset_parameters()
        self.w.reset_parameters()
        self.mlp_final.reset_parameters()

    def forward(self, feats):
        """Forward."""
        m = self.num_nodes
        feat_dim = feats
        row, col = self.g.edges()
        row = row-row.min()
        aa = SparseTensor(
            row=row, col=col, sparse_sizes=(m, m)
                         ).to_torch_sparse_coo_tensor()

        xa = self.mlpa(aa, input_tensor=True)
        xx = self.mlpx(feat_dim, input_tensor=True)
        x = torch.cat((xa, xx), axis=-1)
        x = self.w(x)
        if self.inner_dropout:
            x = F.dropout(x)
        if self.inner_activation:
            x = F.relu(x)
        x = F.relu(x + xa + xx)
        x = self.mlp_final(x, input_tensor=True)

        return x


class MLP(nn.Module):
    """MLP model."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers, dropout=.5):
        """Initiate layer."""
        super().__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        """Reset parameters."""
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, input_tensor=False):
        """Forward."""
        if not input_tensor:
            x = data.graph['node_feat']
        else:
            x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
